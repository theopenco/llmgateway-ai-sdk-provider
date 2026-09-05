import { createLLMGateway } from '@/src';
import { generateText, Output, streamText } from 'ai';
import { expect, it, vi } from 'vitest';
import { z } from 'zod/v4';

vi.setConfig({
  testTimeout: 42_000,
});

const schema = z.object({
  recipe: z
    .object({
      name: z.string().describe('Name of the recipe'),
      ingredients: z
        .array(
          z.object({
            name: z.string().describe('Name of the ingredient'),
            amount: z.string().describe('Amount of the ingredient'),
          }),
        )
        .describe('List of ingredients'),
      steps: z.array(z.string()).describe('Cooking steps'),
    })
    .describe('Recipe details'),
});

it('should generate structured output with json_schema using generateText', async () => {
  const llmgateway = createLLMGateway({
    apiKey: process.env.LLM_GATEWAY_API_KEY,
    baseUrl: process.env.LLM_GATEWAY_API_BASE,
  });
  const model = llmgateway('gpt-4o-mini');

  const result = await generateText({
    model,
    output: Output.object({ schema }),
    prompt: 'Generate a simple recipe for chocolate chip cookies.',
  });

  expect(result.output).toBeDefined();
  expect(result.output.recipe).toBeDefined();
  expect(result.output.recipe.name).toBeDefined();
  expect(typeof result.output.recipe.name).toBe('string');
  expect(Array.isArray(result.output.recipe.ingredients)).toBe(true);
  expect(Array.isArray(result.output.recipe.steps)).toBe(true);
  expect(result.output.recipe.ingredients.length).toBeGreaterThan(0);
  expect(result.output.recipe.steps.length).toBeGreaterThan(0);
});

it('should generate structured output with json_schema using streamText', async () => {
  const llmgateway = createLLMGateway({
    apiKey: process.env.LLM_GATEWAY_API_KEY,
    baseUrl: process.env.LLM_GATEWAY_API_BASE,
  });
  const model = llmgateway('gpt-4o-mini');

  const result = streamText({
    model,
    output: Output.object({ schema }),
    prompt: 'Generate a simple recipe for pancakes.',
  });

  // Consume the stream
  for await (const _partialObject of result.partialOutputStream) {
    // Just consume it
  }

  const finalObject = await result.output;

  expect(finalObject).toBeDefined();
  expect(finalObject.recipe).toBeDefined();
  expect(finalObject.recipe.name).toBeDefined();
  expect(typeof finalObject.recipe.name).toBe('string');
  expect(Array.isArray(finalObject.recipe.ingredients)).toBe(true);
  expect(Array.isArray(finalObject.recipe.steps)).toBe(true);
  expect(finalObject.recipe.ingredients.length).toBeGreaterThan(0);
  expect(finalObject.recipe.steps.length).toBeGreaterThan(0);
});
