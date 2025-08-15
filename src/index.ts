import { genkit, z } from 'genkit';
import { vertexAI, gemini15Pro } from '@genkit-ai/vertexai';
import { startFlowServer } from '@genkit-ai/express';
import { Firestore } from '@google-cloud/firestore';

// --- Initialization ---
const db = new Firestore();

const ai = genkit({
  plugins: [vertexAI({ location: 'australia-southeast1' })],
});

// --- Helper Functions ---
async function getGenkitPromptFromFirestore(promptId: string): Promise<string> {
  try {
    const docRef = db.collection('genkit_prompts').doc(promptId);
    const doc = await docRef.get();
    if (doc.exists) {
      const promptText = doc.data()?.prompt_text;
      if (promptText) {
        console.log(`Successfully fetched prompt '${promptId}' from Firestore.`);
        return promptText;
      }
    }
    throw new Error(`Prompt '${promptId}' not found or is empty.`);
  } catch (error) {
    console.error(`Failed to fetch Genkit prompt '${promptId}':`, error);
    // Provide a safe fallback to prevent crashes
    return "You are a helpful assistant. Your primary prompt failed to load.";
  }
}

// --- ARCE Agent Flows (Pure Google MVP) ---
const architectFlow = ai.defineFlow({ name: 'architectFlow', inputSchema: z.string(), outputSchema: z.string() }, async (taskDescription) => {
    const prompt = await getGenkitPromptFromFirestore('architect');
    const response = await ai.generate({ model: gemini15Pro, messages: [{ role: 'system', content: [{ text: prompt }] }, { role: 'user', content: [{ text: taskDescription }] }], config: { temperature: 0.0 } });
    let plan = response.text;
    if (plan.startsWith("```json")) { plan = plan.slice(7, -3).trim(); }
    try { JSON.parse(plan); return plan; } catch (e) { throw new Error("Architect failed to generate a valid JSON plan."); }
});

const searchAndAnswerFlow = ai.defineFlow({ name: 'searchAndAnswerFlow', inputSchema: z.string(), outputSchema: z.string() }, async (question) => {
    const prompt = await getGenkitPromptFromFirestore('researcher');
    const response = await ai.generate({ model: gemini15Pro, messages: [{ role: 'system', content: [{ text: prompt }] }, { role: 'user', content: [{ text: question }] }], config: { googleSearchRetrieval: {}, maxOutputTokens: 1000 } });
    return response.text;
});

const creatorFlow = ai.defineFlow({ name: 'creatorFlow', inputSchema: z.string(), outputSchema: z.string() }, async (planAndResearch) => {
    const prompt = await getGenkitPromptFromFirestore('creator');
    const response = await ai.generate({ model: gemini15Pro, messages: [{ role: 'system', content: [{ text: prompt }] }, { role: 'user', content: [{ text: planAndResearch }] }], config: { temperature: 0.5 } });
    return response.text;
});

const editorFlow = ai.defineFlow({ name: 'editorFlow', inputSchema: z.string(), outputSchema: z.string() }, async (draft) => {
    const prompt = await getGenkitPromptFromFirestore('editor');
    const response = await ai.generate({ model: gemini15Pro, messages: [{ role: 'system', content: [{ text: prompt }] }, { role: 'user', content: [{ text: draft }] }], config: { temperature: 0.2 } });
    return response.text;
});

// --- Start the Server ---
startFlowServer({
  flows: [
    architectFlow,
    searchAndAnswerFlow,
    creatorFlow,
    editorFlow
  ],
  port: 8080, // Explicitly set the port for Cloud Run
});