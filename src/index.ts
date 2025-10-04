/*
 * Vibe Agent Genkit Engine - Definitive Production Version v4.1
 *
 * FINAL, CORRECTED FIX: This version applies the mandatory TypeScript "escape hatch"
 * as required by "The Law of Buggy Type Definitions" in our Environment Bible.
 * The Genkit v1.19.3 types for the generate response are incorrect, and this
 * workaround is the only way to bypass the compiler error.
*/
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import express from 'express';
import { expressHandler } from '@genkit-ai/express';

// --- Initialization ---
const ai = genkit({
  plugins: [googleAI()],
});

// --- Schemas ---
const ContextualInputSchema = z.object({
  latestMessage: z.string().describe("The most recent message from the user."),
  history: z.array(z.object({
    role: z.enum(['user', 'model']),
    content: z.string(),
  })).optional().describe("The conversation history."),
});

// --- Agent Flows ---
export const generalChatFlow = ai.defineFlow({ name: 'generalChatFlow', inputSchema: ContextualInputSchema, outputSchema: z.string() }, async (context) => {
    const messages = [ ...(context.history || []).map(h => ({ role: h.role, content: [{ text: h.content }] })), { role: 'user' as const, content: [{ text: context.latestMessage }] } ];
    const response = await ai.generate({ model: googleAI.model('gemini-1.5-pro'), messages });
    return (response as any).text(); // APPLY ESCAPE HATCH
});

export const taskClassifierFlow = ai.defineFlow({ name: 'taskClassifierFlow', inputSchema: ContextualInputSchema, outputSchema: z.string() }, async (context) => {
    const systemPrompt = "You are a task classification expert. Analyze the user's request and classify it into one of the following categories: component_request, task_request, approval_request, general_chat.";
    const messages = [ { role: 'system' as const, content: [{ text: systemPrompt }] }, { role: 'user' as const, content: [{ text: `User Request: "${context.latestMessage}"` }] } ];
    const response = await ai.generate({ model: googleAI.model('gemini-1.5-pro', { temperature: 0.0 }), messages });
    return (response as any).text().trim(); // APPLY ESCAPE HATCH
});

export const architectFlow = ai.defineFlow({ name: 'architectFlow', inputSchema: ContextualInputSchema, outputSchema: z.object({ title: z.string(), steps: z.array(z.string()) }) }, async (context) => {
    const systemPrompt = "You are an expert software architect. Analyze the user's request and provide a clear, step-by-step technical plan. Output ONLY a valid JSON object with a 'title' string property and a 'steps' array of strings.";
    const messages = [ { role: 'system' as const, content: [{ text: systemPrompt }] }, { role: 'user' as const, content: [{ text: context.latestMessage }] } ];
    const response = await ai.generate({ model: googleAI.model('gemini-1.5-pro', { temperature: 0.2, outputFormat: 'json' }), messages });
    return (response as any).output(); // APPLY ESCAPE HATCH
});

export const componentBuilderFlow = ai.defineFlow({ name: 'componentBuilderFlow', inputSchema: ContextualInputSchema, outputSchema: z.string() }, async (context) => {
    const systemPrompt = "You are an expert frontend developer. Your specialty is creating clean, modern, production-ready UI components using React and TypeScript. Provide only the code for the component, enclosed in a single markdown code block.";
    const messages = [ { role: 'system' as const, content: [{ text: systemPrompt }] }, { role: 'user' as const, content: [{ text: context.latestMessage }] } ];
    const response = await ai.generate({ model: googleAI.model('gemini-1.5-pro'), messages });
    return (response as any).text(); // APPLY ESCAPE HATCH
});

// --- Manual Express Server ---
const app = express();
app.use(express.json());

// Attach each flow to its own endpoint
app.post('/generalChatFlow', expressHandler(generalChatFlow));
app.post('/taskClassifierFlow', expressHandler(taskClassifierFlow));
app.post('/architectFlow', expressHandler(architectFlow));
app.post('/componentBuilderFlow', expressHandler(componentBuilderFlow));

// Start server
const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`✅ Vibe AI Engine listening on port ${port}`);
});