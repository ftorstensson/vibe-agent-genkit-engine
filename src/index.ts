/*
 * Vibe Agent Genkit Engine - Definitive Production Version v5.0
 * FINAL, VALIDATED FIX: Implements the expert-recommended "Surgical Override"
 * using a manual Express server and the correct `expressHandler` function.
*/
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import express from 'express';
import { expressHandler } from '@genkit-ai/express';

const ai = genkit({ plugins: [googleAI()] });

const ContextualInputSchema = z.object({
  latestMessage: z.string(),
  history: z.array(z.object({ role: z.enum(['user', 'model']), content: z.string() })).optional(),
});

export const generalChatFlow = ai.defineFlow({ name: 'generalChatFlow', inputSchema: ContextualInputSchema, outputSchema: z.string() }, async (context) => {
    const response = await ai.generate({ model: googleAI.model('gemini-1.5-pro'), messages: [ ...(context.history || []).map(h => ({ role: h.role, content: [{ text: h.content }] })), { role: 'user', content: [{ text: context.latestMessage }] } ] });
    return (response as any).text();
});

export const taskClassifierFlow = ai.defineFlow({ name: 'taskClassifierFlow', inputSchema: ContextualInputSchema, outputSchema: z.string() }, async (context) => {
    const systemPrompt = "You are a task classification expert. Classify the user's request into one of the following: component_request, task_request, approval_request, general_chat.";
    const response = await ai.generate({ model: googleAI.model('gemini-1.5-pro', { temperature: 0.0 }), messages: [ { role: 'system', content: [{ text: systemPrompt }] }, { role: 'user', content: [{ text: `User Request: "${context.latestMessage}"` }] } ] });
    return (response as any).text().trim();
});

export const architectFlow = ai.defineFlow({ name: 'architectFlow', inputSchema: ContextualInputSchema, outputSchema: z.object({ title: z.string(), steps: z.array(z.string()) }) }, async (context) => {
    const systemPrompt = "You are an expert software architect. Provide a clear, step-by-step technical plan. Output ONLY a valid JSON object with a 'title' string and a 'steps' array of strings.";
    const response = await ai.generate({ model: googleAI.model('gemini-1.5-pro', { temperature: 0.2, outputFormat: 'json' }), messages: [ { role: 'system', content: [{ text: systemPrompt }] }, { role: 'user', content: [{ text: context.latestMessage }] } ] });
    return (response as any).output();
});

const app = express();
app.use(express.json());

app.post('/generalChatFlow', expressHandler(generalChatFlow));
app.post('/taskClassifierFlow', expressHandler(taskClassifierFlow));
app.post('/architectFlow', expressHandler(architectFlow));

const port = process.env.PORT || 8080;
app.listen(port, () => {
  console.log(`✅ Vibe AI Engine listening on port ${port}`);
});