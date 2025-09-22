/*
 * Milestone 6: Adding the Final Agent Flows
 * Step 6.2: Add the componentBuilderFlow (Complete File)
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

// --- Helper Functions (Commented out) ---
/*
async function getGenkitPromptFromFirestore(promptId: string): Promise<string> {
  try {
    const docRef = db.collection('prompts').doc(promptId);
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
    return response.text.trim();
  }
);

export const architectFlow = ai.defineFlow(
  {
    name: 'architectFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.string(),
  },
  async (context: z.infer<typeof ContextualInputSchema>) => {
    const systemPrompt = "You are an expert software architect. Analyze the user's request and provide a clear, step-by-step technical plan to achieve their goal. The plan should be actionable and easy for a developer to follow.";
    const messages = [
        { role: 'system' as const, content: [{ text: systemPrompt }] },
        { role: 'user' as const, content: [{ text: context.latestMessage }] },
    ];
    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro', { temperature: 0.2 }),
      messages,
    });
    return response.text;
  }
);

export const componentBuilderFlow = ai.defineFlow(
  {
    name: 'componentBuilderFlow',
    inputSchema: ContextualInputSchema,
    outputSchema: z.string(),
  },
  async (context: z.infer<typeof ContextualInputSchema>) => {
    const systemPrompt = "You are an expert frontend developer. Your specialty is creating clean, modern, production-ready UI components using React and TypeScript. Provide only the code for the component, enclosed in a single markdown code block.";
    const messages = [
        { role: 'system' as const, content: [{ text: systemPrompt }] },
        { role: 'user' as const, content: [{ text: context.latestMessage }] },
    ];
    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro'),
      messages,
    });
    return response.text;
  }
);