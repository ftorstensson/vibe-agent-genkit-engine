/*
 * Milestone 5: The Definitive, Working Code with Flow Server
 * This version is a direct transcription of the expert-validated code.
*/
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import { Firestore } from '@google-cloud/firestore';

// --- Initialization ---
const db = new Firestore();

const ai = genkit({
  plugins: [
    googleAI(),
  ],
});

// --- Schemas ---
const ContextualInputSchema = z.object({
  latestMessage: z.string().describe("The most recent message from the user."),
  history: z.array(z.object({
    role: z.enum(['user', 'model']),
    content: z.string(),
  })).optional().describe("The conversation history."),
});

// --- Helper Functions (Commented out for initial testing) ---
/*
async function getGenkitPromptFromFirestore(promptId: string): Promise<string> {
  // ... function code ...
}
*/

// --- Agent Flows ---
export const generalChatFlow = ai.defineFlow(
  {
    name: 'generalChatFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.string(),
  },
  async (context: z.infer<typeof ContextualInputSchema>) => {
    const messages = [
      ...(context.history || []).map(h => ({ role: h.role, content: [{ text: h.content }] })),
      { role: 'user' as const, content: [{ text: context.latestMessage }] },
    ];

    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro'),
      messages,
    });
    
    console.log('generalChatFlow Response object:', response);
    return response.text;
  }
);

export const taskClassifierFlow = ai.defineFlow(
  {
    name: 'taskClassifierFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.string(),
  },
  async (context: z.infer<typeof ContextualInputSchema>) => {
    const systemPrompt = "You are a task classification expert. Analyze the user's request and classify it into one of the following categories: component_request, task_request, approval_request, general_chat.";

    const messages = [
      { role: 'system' as const, content: [{ text: systemPrompt }] },
      { role: 'user' as const, content: [{ text: `User Request: "${context.latestMessage}"` }] },
    ];

    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro', { temperature: 0.0 }),
      messages,
    });

    console.log('taskClassifierFlow Response object:', response);

    const text = response.text;
    return text.trim();
  }
);

