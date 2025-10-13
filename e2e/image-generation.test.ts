import { generateObject, generateText, streamObject, streamText } from 'ai';
import { it, expect, vi } from 'vitest';
import { createLLMGateway } from '@/src';
import { z } from 'zod';

vi.setConfig({
  testTimeout: 60_000,
});

const imageDescriptionSchema = z.object({
  description: z.string().describe('Description of the generated image'),
  style: z.string().describe('Style of the image'),
});

describe('image generation', () => {
  it.skip('should generate an image with description using generateObject', async () => {
    const llmgateway = createLLMGateway({
      apiKey: process.env.LLM_GATEWAY_API_KEY,
      baseUrl: process.env.LLM_GATEWAY_API_BASE,
    });
    const model = llmgateway('gemini-2.5-flash-image-preview');

    const result = await generateObject({
      model,
      schema: imageDescriptionSchema,
      mode: 'json',
      prompt: 'Generate a beautiful sunset over the ocean and describe it',
    });

    expect(result.object).toBeDefined();
    expect(result.object.description).toBeDefined();
    expect(typeof result.object.description).toBe('string');
    expect(result.object.description.length).toBeGreaterThan(0);
    expect(result.object.style).toBeDefined();
    expect(typeof result.object.style).toBe('string');

    // Check for experimental image output
    if (result.experimental_output) {
      expect(result.experimental_output).toBeDefined();
      const outputs = Array.isArray(result.experimental_output)
        ? result.experimental_output
        : [result.experimental_output];

      const imageOutput = outputs.find((output: any) => output.type === 'file');
      if (imageOutput) {
        expect(imageOutput.mediaType).toMatch(/^image\//);
        expect(imageOutput.data).toBeDefined();
        expect(typeof imageOutput.data).toBe('string');
        expect(imageOutput.data.length).toBeGreaterThan(0);
      }
    }
  });

  it.skip('should generate an image with description using streamObject', async () => {
    const llmgateway = createLLMGateway({
      apiKey: process.env.LLM_GATEWAY_API_KEY,
      baseUrl: process.env.LLM_GATEWAY_API_BASE,
    });
    const model = llmgateway('gemini-2.5-flash-image-preview');

    const result = await streamObject({
      model,
      schema: imageDescriptionSchema,
      mode: 'json',
      prompt: 'Generate a serene mountain landscape and describe it',
    });

    // Consume the stream
    for await (const _partialObject of result.partialObjectStream) {
      // Just consume it
    }

    const finalObject = await result.object;

    expect(finalObject).toBeDefined();
    expect(finalObject.description).toBeDefined();
    expect(typeof finalObject.description).toBe('string');
    expect(finalObject.description.length).toBeGreaterThan(0);
    expect(finalObject.style).toBeDefined();
    expect(typeof finalObject.style).toBe('string');

    // Check for experimental image output
    if (result.experimental_output) {
      const outputs = await result.experimental_output;
      expect(outputs).toBeDefined();

      const outputArray = Array.isArray(outputs) ? outputs : [outputs];
      const imageOutput = outputArray.find((output: any) => output.type === 'file');

      if (imageOutput) {
        expect(imageOutput.mediaType).toMatch(/^image\//);
        expect(imageOutput.data).toBeDefined();
        expect(typeof imageOutput.data).toBe('string');
        expect(imageOutput.data.length).toBeGreaterThan(0);
      }
    }
  });

  it('should generate an image using generateText', async () => {
    const llmgateway = createLLMGateway({
      apiKey: process.env.LLM_GATEWAY_API_KEY,
      baseUrl: process.env.LLM_GATEWAY_API_BASE,
    });
    const model = llmgateway('gemini-2.5-flash-image-preview');

    const result = await generateText({
      model,
      prompt: 'Generate a vibrant city skyline at night',
    });

    expect(result.text).toBeDefined();
    expect(typeof result.text).toBe('string');

    // Check for image in response messages
    expect(result.response).toBeDefined();
    expect(result.response.messages).toBeDefined();
    expect(Array.isArray(result.response.messages)).toBe(true);

    // Find the assistant message with image content
    const assistantMessage = result.response.messages.find(
      (msg: any) => msg.role === 'assistant'
    );
    expect(assistantMessage).toBeDefined();

    const imageContent = assistantMessage?.content.find(
      (content: any) => content.type === 'file'
    );

    if (imageContent) {
      expect(imageContent.mediaType).toMatch(/^image\//);
      expect(imageContent.data).toBeDefined();
      expect(typeof imageContent.data).toBe('string');
      expect(imageContent.data.length).toBeGreaterThan(0);

      // Base64 validation - should be a valid base64 string
      const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
      expect(base64Regex.test(imageContent.data)).toBe(true);
    }
  });

  it('should generate an image using streamText', async () => {
    const llmgateway = createLLMGateway({
      apiKey: process.env.LLM_GATEWAY_API_KEY,
      baseUrl: process.env.LLM_GATEWAY_API_BASE,
    });
    const model = llmgateway('gemini-2.5-flash-image-preview');

    const result = await streamText({
      model,
      prompt: 'Generate an abstract colorful painting',
    });

    // Consume the text stream
    let fullText = '';
    for await (const textPart of result.textStream) {
      fullText += textPart;
    }

    expect(fullText).toBeDefined();
    expect(typeof fullText).toBe('string');

    // Wait for the final response
    const response = await result.response;
    expect(response).toBeDefined();
    expect(response.messages).toBeDefined();
    expect(Array.isArray(response.messages)).toBe(true);

    // Find the assistant message with image content
    const assistantMessage = response.messages.find(
      (msg: any) => msg.role === 'assistant'
    );
    expect(assistantMessage).toBeDefined();

    const imageContent = assistantMessage.content.find(
      (content: any) => content.type === 'file'
    );

    if (imageContent) {
      expect(imageContent.mediaType).toMatch(/^image\//);
      expect(imageContent.data).toBeDefined();
      expect(typeof imageContent.data).toBe('string');
      expect(imageContent.data.length).toBeGreaterThan(0);

      // Base64 validation
      const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
      expect(base64Regex.test(imageContent.data)).toBe(true);
    }
  });
})
