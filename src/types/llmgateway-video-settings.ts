/** Any video model ID supported by LLMGateway, including provider prefixes. */
export type LLMGatewayVideoModelId = string;

export type LLMGatewayVideoSettings = {
  /** Additional fields for the gateway's POST /videos endpoint. */
  extraBody?: Record<string, unknown>;
};
