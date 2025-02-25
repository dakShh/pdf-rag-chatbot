import { google } from '@ai-sdk/google';
import { streamText } from 'ai';
import { DataAPIClient } from '@datastax/astra-db-ts';
import { HfInference } from '@huggingface/inference';

const {
    ASTRA_DB_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    ASTRA_DB_NAMESPACE,
    ASTRA_DB_COLLECTION,
    HF_API_TOKEN,
} = process.env;

if (!ASTRA_DB_ENDPOINT || !ASTRA_DB_COLLECTION || !HF_API_TOKEN) {
    throw new Error('Please check your env variables!');
}
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });
const huggingfaceClient = new HfInference(HF_API_TOKEN);

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
    const { messages } = await req.json();
    console.log('messages: ', messages);
    const latestMessage = messages[messages.length - 1]?.content;

    let docContext = '';

    const embedding = await huggingfaceClient.featureExtraction({
        model: 'mixedbread-ai/mxbai-embed-large-v1',
        inputs: latestMessage,
    });

    // Ensure embedding is a number[]
    if (!Array.isArray(embedding)) {
        throw new Error('Expected embedding to be an array of numbers');
    }

    const embeddingArray = embedding as number[];

    try {
        const collection = db.collection(ASTRA_DB_COLLECTION!);
        const cursor = collection.find(
            {},
            {
                sort: { $vector: embeddingArray },
                limit: 10,
            }
        );

        const documents = await cursor.toArray();
        const docsMap = documents?.map((doc) => doc.text);
        docContext = JSON.stringify(docsMap);
    } catch (error) {
        console.log('error: ', error);
    }

    const template = {
        role: 'system',
        content: `You are an AI assistant who only answers questions based on the below context provided, If the context doesn't include the information you need answer politely, "I do not have any information about it.. Please try another question."
        ------------
        START CONTEXT
        ${docContext}
        END CONTEXT
        ------------
        QUESTION: ${latestMessage}
        -------------
        
        `,
    };

    const result = streamText({
        model: google('gemini-1.5-flash'),
        messages: [template, ...messages],
    });

    return result.toDataStreamResponse();
}
