/*
 * File: src/index.ts
 * Version: 5.4.3 (add approval_request to classifier schema)
 * Date: 2025-09-05
 * Objective: Fixes a silent failure in the componentBuilderFlow by refactoring it to use the reliable 'messages' array pattern for AI calls.
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
    const docRef = db.collection('prompts').doc(promptId);
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

// --- Schemas ---
const MessageSchema = z.object({
  role: z.enum(['user', 'model']).describe("The role of the message sender"),
  content: z.string().describe("The text content of the message"),
});

const ChatHistorySchema = z.array(MessageSchema);

const ChatResponseSchema = z.object({
  role: z.literal('model'),
  content: z.string(),
});

const PlanSchema = z.object({
  title: z.string().describe("A short, descriptive title for the plan."),
  steps: z.array(z.string()).describe("A list of clear, actionable steps to accomplish the task."),
});

// --- Agent Flows ---
const projectManagerChatFlow = ai.defineFlow(
  {
    name: 'projectManagerChatFlow',
    inputSchema: ChatHistorySchema,
    outputSchema: ChatResponseSchema,
  },
  async (messages) => {
    const response = await ai.generate({
      model: gemini15Pro,
      system: "You are a helpful and concise project manager AI.",
      messages: messages.map(msg => ({
        role: msg.role,
        content: [{ text: msg.content }],
      })),
      config: { temperature: 0.5 },
    });
    const responseText = response.text;
    if (!responseText) { throw new Error('AI failed to generate a response text.'); }
    return { role: 'model' as const, content: responseText };
  }
);

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
    if (!plan) { throw new Error("Architect failed to generate a valid plan object."); }
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
// === AGENT: Frontend Component Builder (CORRECTED)
// ===================================================================================
const componentBuilderFlow = ai.defineFlow(
  {
    name: 'componentBuilderFlow',
    inputSchema: z.string().describe('A natural language description of a UI component.'),
    outputSchema: z.string().describe('A string containing the generated React/TypeScript component code.'),
  },
  async (description) => {
    const systemPrompt = await getGenkitPromptFromFirestore('pm_component_builder');
    
    // THE FIX: Use the reliable 'messages' array pattern.
    const response = await ai.generate({
      model: gemini15Pro,
      messages: [
        { role: 'system', content: [{ text: systemPrompt }] },
        { role: 'user', content: [{ text: description }] }
      ]
    });
    // END OF FIX

    return response.text;
  }
);
// ===================================================================================

const taskExecutionFlow = ai.defineFlow(
  {
    name: 'taskExecutionFlow',
    inputSchema: z.string().describe('A high-level task description from the user.'),
    outputSchema: z.string().describe('The final, generated artifact (e.g., a report or document).'),
  },
  async (taskDescription) => {
    console.log(`--- Conductor Agent: Starting task '${taskDescription}' ---`);
    const plan = await architectFlow(taskDescription);
    const creatorInput = `
      Please write a document based on the following plan.
      Title: ${plan.title}
      Steps:
      ${plan.steps.map((step, index) => `${index + 1}. ${step}`).join('\n')}
    `;
    const draft = await creatorFlow(creatorInput);
    console.log(`--- Conductor Agent: Task completed successfully! ---`);
    return draft;
  }
);

const taskClassifierFlow = ai.defineFlow(
  {
    name: 'taskClassifierFlow',
    inputSchema: z.string().describe('The user\'s raw text input.'),
    outputSchema: z.enum([
      "component_request",
      "task_request",
      "approval_request",
      "general_chat"
    ]).describe('The classification of the user\'s request.'),
  },
  async (userInput) => {
    const systemPrompt = await getGenkitPromptFromFirestore('pm_task_classifier');
    const response = await ai.generate({
      model: gemini15Pro,
      messages: [
        { role: 'system', content: [{ text: systemPrompt }] },
        { role: 'user', content: [{ text: `User Request: "${userInput}"`}] }
      ],
      config: { temperature: 0.0 }
    });
    const classification = response.text.trim();
    if (["component_request", "task_request", "approval_request", "general_chat"].includes(classification)) {
      return classification as "component_request" | "task_request" | "approval_request" | "general_chat";
    }
    console.warn(`Classifier returned an unexpected value: '${classification}'. Defaulting to 'general_chat'.`);
    return "general_chat";
  }
);

// --- Start the Server ---
startFlowServer({
  flows: [
    projectManagerChatFlow,
    architectFlow,
    searchAndAnswerFlow,
    creatorFlow,
    editorFlow,
    componentBuilderFlow,
    taskExecutionFlow,
    taskClassifierFlow
  ],
  port: 8080,
});