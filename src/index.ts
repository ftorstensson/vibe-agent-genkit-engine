/*
 * Vibe Agent Genkit Engine - Definitive Production Version v2.1
 *
 * SURGICAL FIX: After deployment, it was discovered that the default Genkit
 * `startFlowServer` does not correctly parse JSON request bodies in Cloud Run.
 * The previous attempt to manually build an Express server failed due to a
 * version incompatibility with the `@genkit-ai/express` library.
 *
 * This version uses the less invasive, correct solution: configuring the
 * existing `startFlowServer` with explicit `jsonParserOptions`. This forces
 * the underlying body-parser to have a larger limit and handle our specific
 * JSON payload correctly, fixing the "Provided data: undefined" bug while
 * remaining within the library's intended API for our installed version.
*/
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import { startFlowServer } from '@genkit-ai/express';
import { Express } from 'express'; // Import Express type for configuration

// --- Initialization (Unchanged) ---
const ai = genkit({
  plugins: [
    googleAI(),
  ],
});

// --- Schemas (Unchanged) ---
const ContextualInputSchema = z.object({
  latestMessage: z.string().describe("The most recent message from the user."),
  history: z.array(z.object({
    role: z.enum(['user', 'model']),
    content: z.string(),
  })).optional().describe("The conversation history."),
});

// --- Agent Flows (Unchanged from previous fix) ---
export const generalChatFlow = ai.defineFlow(
  {
    name: 'generalChatFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.string(),
  },
  async (context) => {
    const messages = [
      ...(context.history || []).map(h => ({ role: h.role, content: [{ text: h.content }] })),
      { role: 'user' as const, content: [{ text: context.latestMessage }] },
    ];
    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro'),
      messages,
    });
    return response.text;
  }
);

export const taskClassifierFlow = ai.defineFlow(
  {
    name: 'taskClassifierFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.string(),
  },
  async (context) => {
    const systemPrompt = "You are a task classification expert. Analyze the user's request and classify it into one of the following categories: component_request, task_request, approval_request, general_chat.";
    const messages = [
      { role: 'system' as const, content: [{ text: systemPrompt }] },
      { role: 'user' as const, content: [{ text: `User Request: "${context.latestMessage}"` }] },
    ];
    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro', { temperature: 0.0 }),
      messages,
    });
    return response.text.trim();
  }
);

export const architectFlow = ai.defineFlow(
  {
    name: 'architectFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.object({
      title: z.string().describe("The title of the plan."),
      steps: z.array(z.string()).describe("The steps of the plan."),
    }),
  },
  async (context) => {
    const systemPrompt = "You are an expert software architect. Analyze the user's request and provide a clear, step-by-step technical plan to achieve their goal. The plan should be actionable and easy for a developer to follow. Output ONLY a valid JSON object with a 'title' string property and a 'steps' array of strings.";
    const messages = [
        { role: 'system' as const, content: [{ text: systemPrompt }] },
        { role: 'user' as const, content: [{ text: context.latestMessage }] },
    ];
    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro', { temperature: 0.2, outputFormat: 'json' }),
      messages,
    });
    return response.output();
  }
);

export const componentBuilderFlow = ai.defineFlow(
  {
    name: 'componentBuilderFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.string(),
  },
  async (context) => {
    const systemPrompt = "You are an expert frontend developer. Your specialty is creating clean, modern, production-ready UI components using React and TypeScript. Provide only the code for the component, enclosed in a single markdown code block.";
    const messages = [
        { role: 'system' as const, content: [{ text: context.latestMessage }] },
    ];
    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro'),
      messages,
    });
    return response.text;
  }
);

// --- Production Server Start (MODIFIED WITH EXPLICIT CONFIG) ---
if (process.env.GENKIT_ENV !== 'dev') {
  startFlowServer({
    port: process.env.PORT ? parseInt(process.env.PORT) : 8080,
    flows: [
      generalChatFlow,
      taskClassifierFlow,
      architectFlow,
      componentBuilderFlow
    ],
    // CRITICAL FIX: Explicitly configure the JSON parser.
    jsonParserOptions: {
        limit: '10mb', // Increase the payload size limit
    }
  });
}