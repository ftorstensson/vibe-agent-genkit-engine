/*
 * File: src/index.ts
 * Version: 5.1.0 (Component Builder POC)
 * Date: 2025-08-31
 * Objective: This is the definitive, expert-provided code for the Genkit AI Engine.
 *            It directly implements the validated second opinion to resolve all
 *            known TypeScript errors and establish a stable backend foundation.
 *            This version adds the first proof-of-concept agent, the componentBuilderFlow.
*/

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
    return "You are a helpful assistant. Your primary prompt failed to load.";
  }
}

// --- EXPERT FIX: Corrected Schemas for Chat Flow ---
const MessageSchema = z.object({
  role: z.enum(['user', 'model']).describe("The role of the message sender"),
  content: z.string().describe("The text content of the message"),
});

const ChatHistorySchema = z.array(MessageSchema);

const ChatResponseSchema = z.object({
  role: z.literal('model'),
  content: z.string(),
});

// =================================================================
// --- THE PROJECT MANAGER CHAT FLOW (EXPERT FIX) ---
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
          content: [{ text: msg.content }], // EXPERT SYNTHESIS: Content must be an array of parts
        })),
        config: { temperature: 0.5 },
      });
      
      const responseText = response.text;

      if (!responseText) {
        throw new Error('AI failed to generate a response text.');
      }

      return {
        role: 'model' as const,
        content: responseText,
      };
    } catch (error) {
      console.error('Error in projectManagerChatFlow:', error);
      throw new Error(`Chat flow failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
);

// --- EXPERT FIX: Corrected Existing ARCE Agent Flows ---
const PlanSchema = z.object({
  title: z.string().describe("A short, descriptive title for the plan."),
  steps: z.array(z.string()).describe("A list of clear, actionable steps to accomplish the task."),
});

const architectFlow = ai.defineFlow(
  { 
    name: 'architectFlow', 
    inputSchema: z.string(), 
    outputSchema: PlanSchema 
  },
  async (taskDescription) => {
    const prompt = await getGenkitPromptFromFirestore('architect');
    
    const response = await ai.generate({ 
      model: gemini15Pro,
      system: prompt,
      prompt: taskDescription,
      output: { schema: PlanSchema }, 
      config: { temperature: 0.0 } 
    });
    
    const plan = response.output;
    if (!plan) { 
      throw new Error("Architect failed to generate a valid plan object.");
    }
    return plan;
  }
);

const searchAndAnswerFlow = ai.defineFlow(
  { name: 'searchAndAnswerFlow', inputSchema: z.string(), outputSchema: z.string() }, 
  async (question) => {
    const systemPrompt = await getGenkitPromptFromFirestore('researcher');
    const response = await ai.generate({ model: gemini15Pro, system: systemPrompt, prompt: question, config: { googleSearchRetrieval: {}, maxOutputTokens: 1000 } });
    return response.text;
  }
);

const creatorFlow = ai.defineFlow(
  { name: 'creatorFlow', inputSchema: z.string(), outputSchema: z.string() }, 
  async (planAndResearch) => {
    const systemPrompt = await getGenkitPromptFromFirestore('creator');
    const response = await ai.generate({ model: gemini15Pro, system: systemPrompt, prompt: planAndResearch, config: { temperature: 0.5 } });
    return response.text;
  }
);

const editorFlow = ai.defineFlow(
  { name: 'editorFlow', inputSchema: z.string(), outputSchema: z.string() }, 
  async (draft) => {
    const systemPrompt = await getGenkitPromptFromFirestore('editor');
    const response = await ai.generate({ model: gemini15Pro, system: systemPrompt, prompt: draft, config: { temperature: 0.2 } });
    return response.text;
  }
);


// ===================================================================================
// === NEW AGENT: Frontend Component Builder (Proof of Concept)
// ===================================================================================
const componentBuilderFlow = ai.defineFlow(
  {
    name: 'componentBuilderFlow',
    inputSchema: z.string().describe('A natural language description of a UI component.'),
    outputSchema: z.string().describe('A string containing the generated React/TypeScript component code.'),
  },
  async (description) => {
    
    // For now, we will hardcode the prompt. In the future, we can move this to Firestore.
    const systemPrompt = `
      You are an expert frontend developer specializing in React and TypeScript.
      Your task is to generate a single, self-contained, production-quality React component based on the user's request.

      RULES:
      1.  **Output ONLY the TypeScript code** for the .tsx file.
      2.  Do NOT include any explanations, apologies, or introductory sentences.
      3.  Do NOT wrap the code in markdown backticks (\`\`\`).
      4.  The component must be a modern, functional React component using hooks.
      5.  Use Tailwind CSS for styling.
      6.  The code must be complete and syntactically correct.
    `;
    
    const response = await ai.generate({
      model: gemini15Pro,
      system: systemPrompt,
      prompt: description,
    });

    return response.text;
  }
);
// ===================================================================================


// --- Start the Server ---
startFlowServer({
  flows: [
    projectManagerChatFlow,
    architectFlow,
    searchAndAnswerFlow,
    creatorFlow,
    editorFlow,
    componentBuilderFlow // <-- NEW FLOW ADDED HERE
  ],
  port: 8080,
});