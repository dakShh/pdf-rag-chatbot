import { HfInference } from '@huggingface/inference';

import 'dotenv/config';

const { HF_API_TOKEN } = process.env;

if (!HF_API_TOKEN) {
    throw new Error('Please check hugging face token');
}

const client = new HfInference(HF_API_TOKEN);

const main = async () => {
    const output = await client.featureExtraction({
        model: 'mixedbread-ai/mxbai-embed-large-v1',
        inputs: 'Today is a sunny day and I will get some ice cream.',
    });
    console.log(output);

    return output;
};

main();
