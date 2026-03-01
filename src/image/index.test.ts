import { createLLMGateway } from '../provider';
import { createTestServer } from '../tests/create-test-server';

const TEST_BASE64_IMAGE =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

const testUrl = 'https://test.llmgateway.io/v1/images/generations';
const testEditsUrl = 'https://test.llmgateway.io/v1/images/edits';

const server = createTestServer({
  [testUrl]: {
    response: {
      type: 'json-value',
      body: {
        data: [{ b64_json: TEST_BASE64_IMAGE }],
      },
    },
  },
  [testEditsUrl]: {
    response: {
      type: 'json-value',
      body: {
        data: [{ b64_json: TEST_BASE64_IMAGE }],
      },
    },
  },
});

const provider = createLLMGateway({
  apiKey: 'test-api-key',
  baseURL: 'https://test.llmgateway.io/v1',
  compatibility: 'strict',
  fetch: server.fetch,
});

const model = provider.image('qwen-image-plus');

describe('LLMGatewayImageModel', () => {
  beforeEach(() => {
    server.calls.length = 0;
  });

  describe('doGenerate', () => {
    it('should send correct request body', async () => {
      await model.doGenerate({
        prompt: 'a cute cat',
        n: 1,
        size: '1024x1024',
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      const body = await server.calls[0]!.requestBodyJson;
      expect(body).toEqual({
        model: 'qwen-image-plus',
        prompt: 'a cute cat',
        n: 1,
        size: '1024x1024',
        response_format: 'b64_json',
      });
    });

    it('should return base64 image data', async () => {
      const result = await model.doGenerate({
        prompt: 'a cute cat',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.images).toEqual([TEST_BASE64_IMAGE]);
    });

    it('should not include size in body when undefined', async () => {
      await model.doGenerate({
        prompt: 'a sunset',
        n: 2,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      const body = await server.calls[0]!.requestBodyJson;
      expect(body).toEqual({
        model: 'qwen-image-plus',
        prompt: 'a sunset',
        n: 2,
        response_format: 'b64_json',
      });
    });

    it('should send aspect_ratio in body when provided', async () => {
      await model.doGenerate({
        prompt: 'a dog',
        n: 1,
        size: undefined,
        aspectRatio: '16:9',
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      const body = await server.calls[0]!.requestBodyJson;
      expect(body).toEqual({
        model: 'qwen-image-plus',
        prompt: 'a dog',
        n: 1,
        response_format: 'b64_json',
        aspect_ratio: '16:9',
      });
    });

    it('should warn for unsupported seed', async () => {
      const result = await model.doGenerate({
        prompt: 'a dog',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: 42,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.warnings).toContainEqual({
        type: 'unsupported',
        feature: 'seed',
      });
    });

    it('should send authorization header', async () => {
      await model.doGenerate({
        prompt: 'a tree',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(server.calls[0]!.requestHeaders.authorization).toBe(
        'Bearer test-api-key',
      );
    });

    it('should return multiple images', async () => {
      const multiServer = createTestServer({
        [testUrl]: {
          response: {
            type: 'json-value',
            body: {
              data: [
                { b64_json: TEST_BASE64_IMAGE },
                { b64_json: TEST_BASE64_IMAGE },
              ],
            },
          },
        },
      });

      const multiModel = createLLMGateway({
        apiKey: 'test-api-key',
        baseURL: 'https://test.llmgateway.io/v1',
        compatibility: 'strict',
        fetch: multiServer.fetch,
      }).image('qwen-image-plus');

      const result = await multiModel.doGenerate({
        prompt: 'two cats',
        n: 2,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.images).toHaveLength(2);
    });

    it('should include response metadata', async () => {
      const result = await model.doGenerate({
        prompt: 'a bird',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(result.response.modelId).toBe('qwen-image-plus');
      expect(result.response.timestamp).toBeInstanceOf(Date);
    });

    it('should set specificationVersion to v3', () => {
      expect(model.specificationVersion).toBe('v3');
    });

    it('should set provider to llmgateway.image', () => {
      expect(model.provider).toBe('llmgateway.image');
    });

    it('should send to /images/edits when files are provided', async () => {
      await model.doGenerate({
        prompt: 'add a hat',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: [
          {
            type: 'file',
            mediaType: 'image/png',
            data: TEST_BASE64_IMAGE,
          },
        ],
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(server.calls[0]!.url).toBe(testEditsUrl);
      const body = await server.calls[0]!.requestBodyJson;
      expect(body).toEqual({
        model: 'qwen-image-plus',
        prompt: 'add a hat',
        n: 1,
        response_format: 'b64_json',
        images: [
          {
            image_url: `data:image/png;base64,${TEST_BASE64_IMAGE}`,
          },
        ],
      });
    });

    it('should send to /images/edits with URL-type files', async () => {
      await model.doGenerate({
        prompt: 'make it blue',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: [
          {
            type: 'url',
            url: 'https://example.com/image.png',
          },
        ],
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(server.calls[0]!.url).toBe(testEditsUrl);
      const body = await server.calls[0]!.requestBodyJson;
      expect(body).toEqual({
        model: 'qwen-image-plus',
        prompt: 'make it blue',
        n: 1,
        response_format: 'b64_json',
        images: [
          {
            image_url: 'https://example.com/image.png',
          },
        ],
      });
    });

    it('should send to /images/edits with Uint8Array file data', async () => {
      const binaryData = new Uint8Array([137, 80, 78, 71]);
      await model.doGenerate({
        prompt: 'edit this',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: [
          {
            type: 'file',
            mediaType: 'image/png',
            data: binaryData,
          },
        ],
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(server.calls[0]!.url).toBe(testEditsUrl);
      const body = await server.calls[0]!.requestBodyJson;
      expect(body.images).toEqual([
        {
          image_url: `data:image/png;base64,${Buffer.from(binaryData).toString('base64')}`,
        },
      ]);
    });

    it('should send to /images/generations when files is undefined', async () => {
      await model.doGenerate({
        prompt: 'a cat',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: undefined,
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(server.calls[0]!.url).toBe(testUrl);
    });

    it('should send to /images/generations when files is empty', async () => {
      await model.doGenerate({
        prompt: 'a cat',
        n: 1,
        size: undefined,
        aspectRatio: undefined,
        seed: undefined,
        files: [],
        mask: undefined,
        providerOptions: {},
        headers: {},
      });

      expect(server.calls[0]!.url).toBe(testUrl);
    });
  });
});
