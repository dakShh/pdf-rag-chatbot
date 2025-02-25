/* eslint-disable @typescript-eslint/no-unused-vars */

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
// Or, in web environments:
// import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";
// const blob = new Blob(); // e.g. from a file input
// const loader = new WebPDFLoader(blob);

import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { DataAPIClient } from '@datastax/astra-db-ts';
import { HfInference } from '@huggingface/inference';

import 'dotenv/config';

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

const PATH: string = 'src/documents/report.pdf';
const batchsize = 10;

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });
const huggingfaceClient = new HfInference(HF_API_TOKEN);

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 35,
});

async function loadPdfContent(path: string) {
    const loader = new PDFLoader(path, {
        splitPages: false,
    });
    const docs = await loader.load();
    return docs;
}

type SimilarityMetric = 'dot_product' | 'cosine' | 'euclidean';
const createCollection = async (similarityMetric: SimilarityMetric = 'dot_product') => {
    const res = await db.createCollection(ASTRA_DB_COLLECTION, {
        vector: {
            dimension: 1024,
            metric: similarityMetric,
        },
    });

    console.log('Collection created: ', res);
};

const loadDatatToVectorDatabase = async () => {
    console.log('Loading sample data..');
    const collection = db.collection(ASTRA_DB_COLLECTION);
    console.log('Collection name: ', collection);

    try {
        const content = await loadPdfContent(PATH);

        console.log('Content Length: ', content[0].pageContent.length);
        const chunks = await splitter.splitText(content[0].pageContent);
        console.log('Chunk size: ', chunks.length);

        // let totalDocumentChunks = chunks.length;
        // let totalDocumentChunksUpseted = 0;

        console.log('Embedding chunks..');
        for await (const chunk of chunks) {
            console.log('Start new chunk..');
            // const embedding = await openai.embeddings.create({
            //     model: 'text-embedding-3-small',
            //     input: chunk,
            //     encoding_format: 'float',
            // });

            // const vector = embedding.data[0];

            // const res = await collection.insertOne({
            //     $vector: vector,
            //     text: chunk,
            // });

            const embedding = await huggingfaceClient.featureExtraction({
                model: 'mixedbread-ai/mxbai-embed-large-v1',
                inputs: chunk,
            });

            const vector = embedding;

            const res = await collection.insertOne({
                $vector: vector,
                text: chunk,
            });

            console.log('Chunk : ', res);
            console.log('-------------------------------------------------------');
        }
    } catch (error) {
        console.log('Error: ', error);
    }
};
createCollection().then(() => loadDatatToVectorDatabase());
