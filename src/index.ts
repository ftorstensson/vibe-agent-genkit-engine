/*
 * File: src/index.ts
 * Version: 5.3.1 (Task Classifier Agent - FIX)
 * Date: 2025-09-04
 * Objective: Fixes a TypeError in the taskClassifierFlow by restructuring the
 *            ai.generate() call to use the 'messages' array correctly.
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
// ... [No changes to helper functions] ...
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

// --- Schemas ---
// ... [No changes to schemas] ...
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

// --- Existing Agent Flows ---
// ... [No changes to other flows] ...
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
          content: [{ text: msg.content }],
        })),
        config: { temperature: 0.5 },
      });
      const responseText = response.text;
      if (!responseText) { throw new Error('AI failed to generate a response text.'); }
      return { role: 'model' as const, content: responseText };
    } catch (error) {
      console.error('Error in projectManagerChatFlow:', error);
      throw new Error(`Chat flow failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
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

const componentBuilderFlow = ai.defineFlow(
  {
    name: 'componentBuilderFlow',
    inputSchema: z.string().describe('A natural language description of a UI component.'),
    outputSchema: z.string().describe('A string containing the generated React/TypeScript component code.'),
  },
  async (description) => {
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
    const response = await ai.generate({ model: gemini15Pro, system: systemPrompt, prompt: description });
    return response.text;
  }
);

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
      Ensure the final output is well-structured and comprehensive.
      Title: ${plan.title}
      Steps:
      ${plan.steps.map((step, index) => `${index + 1}. ${step}`).join('\n')}
    `;
    const draft = await creatorFlow(creatorInput);
    console.log(`--- Conductor Agent: Task completed successfully! ---`);
    return draft;
  }
);

// ===================================================================================
// === META AGENT: Task Classifier (For Smart Router) - CORRECTED
// ===================================================================================
const taskClassifierFlow = ai.defineFlow(
  {
    name: 'taskClassifierFlow',
    inputSchema: z.string().describe('The user\'s raw text input.'),
    outputSchema: z.enum([
      "component_request", // For requests to build a UI component
      "task_request",      // For complex tasks requiring a multi-agent team
      "general_chat"       // For anything else
    ]).describe('The classification of the user\'s request.'),
  },
  async (userInput) => {
    const systemPrompt = `
      You are an expert task classifier. Your job is to analyze the user's request and classify it into one of the following categories.
      You must respond with ONLY the category name and nothing else.

      Categories:
      - "component_request": The user is asking to create, build, design, or make a visual UI component. Keywords: "button", "form", "card", "navbar", "component", "UI", "design".
      - "task_request": The user is asking for a complex, multi-step task to be completed. Keywords: "write", "draft", "create a report", "summarize", "plan", "analyze".
      - "general_chat": The user is making a simple conversational statement, asking a question, or saying hello.
    `;
    
    // --- THE FIX ---
    // We must use the 'messages' array and provide both a system and a user role.
    const response = await ai.generate({
      model: gemini15Pro,
      messages: [
        { role: 'system', content: [{ text: systemPrompt }] },
        { role: 'user', content: [{ text: `User Request: "${userInput}"`}] }
      ],
      output: { schema: z.string() }, // This is fine, we still want a simple string
      config: { temperature: 0.0 }
    });
    // --- END OF FIX ---

    // Validate the response is one of the allowed enums, default to general_chat if not.
    const classification = response.text;
    if (["component_request", "task_request", "general_chat"].includes(classification)) {
      return classification as "component_request" | "task_request" | "general_chat";
    }
    
    console.warn(`Classifier returned an unexpected value: '${classification}'. Defaulting to 'general_chat'.`);
    return "general_chat";
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
    componentBuilderFlow,
    taskExecutionFlow,
    taskClassifierFlow
  ],
  port: 8080,
});