/*
 * Milestone 4: The First Real Flow - DEFINITIVE CORRECTION 2
 * Objective: A direct transcription of the expert-validated patterns I previously failed to implement.
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
    // This is the proper structure for the messages array.
    const messages = [
      // CORRECTED: Added 'as ('user' | 'model')' to provide a specific type assertion.
      ...(context.history || []).map(h => ({ role: h.role as ('user' | 'model'), content: [{ text: h.content }] })),
      { role: 'user' as const, content: [{ text: context.latestMessage }] },
    ];

    const response = await ai.generate({
      model: googleAI.model('gemini-1.5-pro'),
      messages,
    });

    return response.text;
  }
);