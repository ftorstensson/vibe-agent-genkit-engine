import { genkit, z } from 'genkit';
import { vertexAI } from '@genkit-ai/vertexai';
import { startFlowServer } from '@genkit-ai/express';

// Initialize Genkit with the Vertex AI plugin
const ai = genkit({
  plugins: [vertexAI()],
});

// Define a simple "hello world" flow using the correct syntax
const helloFlow = ai.defineFlow(
  {
    name: 'helloFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (name) => {
    return `Hello, ${name}!`;
  }
);

// Start the Express server, passing the defined flows
startFlowServer({
  flows: [helloFlow],
});