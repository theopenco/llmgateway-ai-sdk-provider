import type {
  Experimental_VideoModelV4,
  Experimental_VideoModelV4CallOptions,
  Experimental_VideoModelV4File,
  Experimental_VideoModelV4OperationStartResult,
  Experimental_VideoModelV4OperationStatusResult,
  SharedV4Warning,
} from '@ai-sdk/provider';
import type {
  LLMGatewayVideoModelId,
  LLMGatewayVideoSettings,
} from '../types/llmgateway-video-settings';

import {
  APICallError,
  InvalidArgumentError,
  InvalidResponseDataError,
  UnsupportedFunctionalityError,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  convertUint8ArrayToBase64,
  createBinaryResponseHandler,
  createJsonResponseHandler,
  getFromApi,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';

import { llmgatewayFailedResponseHandler } from '../schemas/error-response';

type LLMGatewayVideoConfig = {
  provider: string;
  headers: () => Record<string, string | undefined>;
  url: (options: { modelId: string; path: string }) => string;
  fetch?: typeof fetch;
  extraBody?: Record<string, unknown>;
};

const videoResponseSchema = z.object({
  id: z.string().min(1),
  model: z.string(),
  status: z.enum([
    'queued',
    'in_progress',
    'completed',
    'failed',
    'canceled',
    'expired',
  ]),
  error: z
    .object({ code: z.string().optional(), message: z.string() })
    .nullish(),
});

function fileUrl(file: Experimental_VideoModelV4File): string {
  if (file.type === 'url') {
    return file.url;
  }
  const data =
    typeof file.data === 'string'
      ? file.data
      : convertUint8ArrayToBase64(file.data);
  return `data:${file.mediaType};base64,${data}`;
}

export class LLMGatewayVideoModel implements Experimental_VideoModelV4 {
  readonly specificationVersion = 'v4' as const;
  readonly maxVideosPerCall = 1;

  constructor(
    readonly modelId: LLMGatewayVideoModelId,
    readonly settings: LLMGatewayVideoSettings,
    private readonly config: LLMGatewayVideoConfig,
  ) {}

  get provider(): string {
    return this.config.provider;
  }

  async doStart(
    options: Experimental_VideoModelV4CallOptions & { webhookUrl?: string },
  ): Promise<Experimental_VideoModelV4OperationStartResult> {
    const warnings: SharedV4Warning[] = [];
    if (options.webhookUrl != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'webhookUrl',
        details:
          'Configure signed callbacks with callback_url and callback_secret in providerOptions.llmgateway.',
      });
    }
    for (const feature of ['aspectRatio', 'fps', 'seed'] as const) {
      if (options[feature] != null) {
        warnings.push({
          type: 'unsupported',
          feature,
          details:
            feature === 'aspectRatio'
              ? 'Use resolution to select the video size and aspect ratio.'
              : `LLMGateway does not support ${feature} for video generation.`,
        });
      }
    }

    const body: Record<string, unknown> = {
      model: this.modelId,
      prompt: options.prompt,
      seconds: options.duration,
      size: options.resolution,
      audio: options.generateAudio,
    };
    for (const frame of options.frameImages ?? []) {
      const field = frame.frameType === 'first_frame' ? 'image' : 'last_frame';
      if (body[field] != null) {
        throw new InvalidArgumentError({
          argument: 'frameImages',
          message: `Only one ${frame.frameType} can be provided.`,
        });
      }
      body[field] = { image_url: fileUrl(frame.image) };
    }
    // The SDK also exposes first_frame as options.image for providers that
    // only implement image-to-video. Avoid treating that alias as a duplicate.
    if (options.image != null && body.image == null) {
      body.image = { image_url: fileUrl(options.image) };
    }

    const referenceImages: { image_url: string }[] = [];
    const referenceVideos: { video_url: string }[] = [];
    for (const reference of options.inputReferences ?? []) {
      if (reference.mediaType?.startsWith('video/')) {
        if (reference.type !== 'url' || !reference.url.startsWith('https://')) {
          throw new UnsupportedFunctionalityError({
            functionality: 'video references other than HTTPS URLs',
          });
        }
        referenceVideos.push({ video_url: reference.url });
      } else {
        referenceImages.push({ image_url: fileUrl(reference) });
      }
    }
    if (referenceImages.length > 0) {
      body.reference_images = referenceImages;
    }
    if (referenceVideos.length > 0) {
      body.reference_videos = referenceVideos;
    }

    const requestBody = {
      ...body,
      ...this.config.extraBody,
      ...this.settings.extraBody,
      ...options.providerOptions.llmgateway,
    };
    if (options.n !== 1 || (requestBody.n != null && requestBody.n !== 1)) {
      throw new UnsupportedFunctionalityError({
        functionality:
          'multiple videos per API call; use generateVideo with n instead',
      });
    }
    if (
      typeof requestBody.seconds !== 'number' ||
      !Number.isInteger(requestBody.seconds) ||
      requestBody.seconds < 1
    ) {
      throw new InvalidArgumentError({
        argument: 'duration',
        message:
          'LLMGateway requires a positive integer duration in seconds supported by the selected video model.',
      });
    }
    if (
      typeof requestBody.prompt !== 'string' ||
      requestBody.prompt.length === 0
    ) {
      throw new InvalidArgumentError({
        argument: 'prompt',
        message:
          'LLMGateway requires a text prompt, including for image-to-video generation.',
      });
    }

    const url = this.config.url({ modelId: this.modelId, path: '/videos' });
    const { value: job, responseHeaders } = await postJsonToApi({
      url,
      headers: combineHeaders(this.config.headers(), options.headers),
      body: requestBody,
      failedResponseHandler: llmgatewayFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(videoResponseSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });
    if (['failed', 'canceled', 'expired'].includes(job.status)) {
      throw new APICallError({
        message: job.error?.message ?? `Video generation ${job.status}.`,
        url,
        requestBodyValues: requestBody,
        isRetryable: false,
        data: job,
      });
    }
    return {
      operation: job.id,
      warnings,
      response: {
        timestamp: new Date(),
        modelId: job.model,
        headers: responseHeaders,
      },
    };
  }

  async doStatus({
    operation,
    abortSignal,
    headers,
  }: Parameters<
    NonNullable<Experimental_VideoModelV4['doStatus']>
  >[0]): Promise<Experimental_VideoModelV4OperationStatusResult> {
    if (typeof operation !== 'string' || operation.length === 0) {
      throw new InvalidArgumentError({
        argument: 'operation',
        message: 'Expected a video job ID returned by doStart.',
      });
    }
    const url = this.config.url({
      modelId: this.modelId,
      path: `/videos/${encodeURIComponent(operation)}`,
    });
    const requestOptions = {
      headers: combineHeaders(this.config.headers(), headers),
      failedResponseHandler: llmgatewayFailedResponseHandler,
      abortSignal,
      fetch: this.config.fetch,
      validateUrl: false,
    };
    const { value: job, responseHeaders } = await getFromApi({
      ...requestOptions,
      url,
      successfulResponseHandler: createJsonResponseHandler(videoResponseSchema),
    });
    const response = {
      timestamp: new Date(),
      modelId: job.model,
      headers: responseHeaders,
    };
    if (job.status === 'queued' || job.status === 'in_progress') {
      return { status: 'pending', response };
    }
    const providerMetadata = {
      llmgateway: { videos: [{ id: job.id, status: job.status }] },
    };
    if (job.status !== 'completed') {
      return {
        status: 'error',
        error: job.error?.message ?? `Video generation ${job.status}.`,
        providerMetadata,
        response,
      };
    }

    // Download through the authenticated gateway endpoint. Upstream content
    // URLs can require provider credentials or expire independently of the job.
    const { value: data, responseHeaders: contentHeaders } = await getFromApi({
      ...requestOptions,
      url: `${url}/content`,
      successfulResponseHandler: createBinaryResponseHandler(),
    });
    const contentType =
      contentHeaders?.['content-type']?.split(';')[0]?.trim() || 'video/mp4';
    const mediaType =
      contentType === 'application/octet-stream' ? 'video/mp4' : contentType;
    if (data.length === 0 || !mediaType.startsWith('video/')) {
      throw new InvalidResponseDataError({
        message: 'Expected non-empty video content from LLMGateway.',
        data: { mediaType, byteLength: data.length },
      });
    }
    return {
      status: 'completed',
      videos: [{ type: 'binary', data, mediaType }],
      warnings: [],
      providerMetadata,
      response,
    };
  }
}
