/*
 * File: src/index.ts
 * Version: 4.0.0 (Definitive Synthesis)
 * Date: 2025-08-29
 * Objective: This is the final, correct version of the Genkit AI Engine service.
 *            It synthesizes all expert advice and documentation to resolve all
 *            known TypeScript errors. It is the stable foundation for our backend.
*/

import { genkit, z } from 'genkit';
import { vertexAI, gemini15Pro } from '@genkit-ai/vertexai';
import { startFlowServer } from '@genkit-ai/express';
import { Firestore } from '@google-cloud/firestore';
import { GenkitError } from 'genkit/lib/error';

// --- Initialization ---
const db = new Firestore();
const ai = genkit({
  plugins: [vertexAI({ location: 'australia-southeast1' })],
});

// --- Helper Functions (Unchanged) ---
async function getGenkitPromptFromFirestore(promptId: string): Promise<string> {
  // ... (code is correct, no changes needed)
  try {
    const docRef = db.collection('genkit_prompts').doc(promptId);
    const doc = await docRef.get();
    if (doc.exists) {
      const promptText = doc.data()?.prompt_text;
      if (promptText) { return promptText; }
    }
    throw new Error(`Prompt '${promptId}' not found or is empty.`);
  } catch (error) {
    console.error(`Failed to fetch Genkit prompt '${promptId}':`, error);
    return "You are a helpful assistant.";
  }
}

// --- Schemas for Chat Flow ---
const MessageSchema = z.object({
  role: z.enum(['user', 'model']),
  content: z.string(),
});

const ChatHistorySchema = z.array(MessageSchema);
const ChatResponseSchema = z.object({
  role: z.literal('model'),
  content: z.string(),
});

// =================================================================
// --- THE PROJECT MANAGER CHAT FLOW (DEFINITIVE SYNTHESIS FIX) ---
// =================================================================
const projectManagerChatFlow = ai.defineFlow(
  {
    name: 'projectManagerChatFlow',
    inputSchema: ChatHistorySchema,
    outputSchema: ChatResponseSchema,
  },
  async (messages) => {
    try {
      const response = await ai.generate({
        model: gemini15Pro,
        system: "You are a helpful and concise project manager AI.",
        messages: messages.map(msg => ({
          role: msg.role,
          // --- DEFINITIVE FIX: 'content' MUST be an array of parts ---
          content: [{ text: msg.content }]
        })),
        config: { temperature: 0.5 },
      });
      
      const responseText = response.text();

      if (!responseText) {
        throw new Error('AI failed to generate a response text.');
      }

      return {
        role: 'model' as const,
        content: responseText,
      };
    } catch (error) {
      console.error('Error in projectManagerChatFlow:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new GenkitError({
          status: 'INTERNAL',
          message: `Chat flow failed: ${errorMessage}`
      });
    }
  }
);


// --- ARCE Agent Flows (with definitive .output() fix) ---
const PlanSchema = z.object({
  title: z.string().describe("A short, descriptive title for the plan."),
  steps: z.array(z.string()).describe("A list of clear, actionable steps to accomplish the task."),
});

const architectFlow = ai.defineFlow(
  { name: 'architectFlow', inputSchema: z.string(), outputSchema: PlanSchema },
  async (taskDescription) => {
    const prompt = await getGenkitPromptFromFirestore('architect');
    const response = await ai.generate({ model: gemini15Pro, system: prompt, prompt: taskDescription, output: { schema: PlanSchema }, config: { temperature: 0.0 } });
    const plan = response.output();
    if (!plan) { throw new Error("Architect failed to generate a valid plan object."); }
    return plan;
  }
);

// ... (Other ARCE flows remain the same, but with .text() fix)
const searchAndAnswerFlow = ai.defineFlow({ name: 'searchAndAnswerFlow', inputSchema: z.string(), outputSchema: z.string() }, async (question) => {
    const systemPrompt = await getGenkitPromptFromFirestore('researcher');
    const response = await ai.generate({ model: gemini15Pro, system: systemPrompt, prompt: question, config: { googleSearchRetrieval: {}, maxOutputTokens: 1000 } });
    return response.text();
});
const creatorFlow = ai.defineFlow({ name: 'creatorFlow', inputSchema: z.string(), outputSchema: z.string() }, async (planAndResearch) => {
    const systemPrompt = await getGenkitPromptFromFirestore('creator');
    const response = await ai.generate({ model: gemini15Pro, system: systemPrompt, prompt: planAndResearch, config: { temperature: 0.5 } });
    return response.text();
});
const editorFlow = ai.defineFlow({ name: 'editorFlow', inputSchema: z.string(), outputSchema: z.string() }, async (draft) => {
    const systemPrompt = await getGenkitPromptFromFirestore('editor');
    const response = await ai.generate({ model: gemini15Pro, system: systemPrompt, prompt: draft, config: { temperature: 0.2 } });
    return response.text();
});

// --- Start the Server ---
startFlowServer({
  flows: [
    projectManagerChatFlow,
    architectFlow,
    searchAndAnswerFlow,
    creatorFlow,
    editorFlow
  ],
  port: 8080,
});