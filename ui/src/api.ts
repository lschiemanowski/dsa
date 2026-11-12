export type ChatStatus =
  | "open"
  | "clarifying"
  | "awaiting_decision"
  | "running"
  | "succeeded"
  | "failed";

export interface Proposal {
  id: string;
  digest: string;
  expires_at: string;
  request_json: string;
  payload: {
    question: string;
    interpretation: {
      measure: string;
      population: string;
      group_by: string[];
      filters: string[];
      time_window: string | null;
      units: string | null;
    };
    analysis_guidance?: string;
    allow_plots?: boolean;
  };
}

export interface Notebook {
  url: string;
  sha256: string;
  byte_length: number;
}

export interface Result {
  run_id: string;
  proposal_id: string;
  response_json: string | null;
  notebook: Notebook | null;
}

export interface Conversation {
  id: string;
  welcome: string;
  status: ChatStatus;
  messages: {
    role: "user" | "assistant";
    text: string;
    proposal: Proposal | null;
  }[];
  pending_proposal_id: string | null;
  result: Result | null;
  failure_code: string | null;
}

export type Decision = "approve" | "revise";

const conversations = "/api/conversations";
const conversationPath = (id: string) =>
  conversations + "/" + encodeURIComponent(id);

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
  ) {
    super(code);
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    cache: "no-store",
    ...options,
  });
  if (!response.ok) {
    const error: { detail?: unknown } | null = await response
      .json()
      .catch(() => null);
    throw new ApiError(
      response.status,
      typeof error?.detail === "string" ? error.detail : "request_failed",
    );
  }
  try {
    return await response.json();
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new ApiError(response.status, "invalid_response");
    }
    throw error;
  }
}

function post<T>(path: string, body: unknown) {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function createConversation(signal?: AbortSignal) {
  return request<Conversation>(conversations, { method: "POST", signal });
}

export function readConversation(id: string, signal?: AbortSignal) {
  return request<Conversation>(conversationPath(id), { signal });
}

export function sendMessage(id: string, text: string) {
  return post<Conversation>(conversationPath(id) + "/messages", { text });
}

export function decide(id: string, proposal: Proposal, decision: Decision) {
  const route = "/proposals/" + encodeURIComponent(proposal.id) + "/decision";
  return post<Conversation>(conversationPath(id) + route, {
    decision,
    digest: proposal.digest,
  });
}

export function readNotebook(
  id: string,
  signal: AbortSignal,
): Promise<unknown> {
  return request(conversationPath(id) + "/notebook", { signal });
}

export function explainError(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.code) {
      case "conversation_not_found":
        return "This conversation has expired. Start a new conversation to continue.";
      case "browser_conversation_limit":
        return "This browser has reached its conversation limit. You can continue the current conversation or try again after an idle conversation expires.";
      case "conversation_capacity_reached":
        return "The chat server has reached its conversation limit. Try again after an idle conversation expires.";
      case "analysis_capacity_reached":
        return "Both analysis slots are busy. Try running this proposal again shortly.";
      case "proposal_expired":
      case "proposal_not_pending":
        return "This proposal is no longer awaiting review.";
      case "proposal_pending":
        return "Review the pending proposal before sending another question.";
      case "conversation_busy":
        return "The clarifier is already working on a question.";
      case "conversation_closed":
        return "This conversation is complete. Start a new conversation to ask another question.";
      case "invalid_origin":
      case "invalid_host":
        return "The chat API rejected this address. Open the chat using the configured local URL.";
      case "proposal_digest_mismatch":
        return "The proposal no longer matches the server’s copy. Reload the conversation before approving.";
      case "invalid_response":
        return "The chat server returned an unreadable response. Try again.";
      case "invalid_request":
        return "The server could not accept this input. Check its length and try again.";
    }
    return error.status >= 500
      ? "The chat server could not complete the request. Try again."
      : "The chat server rejected the request (HTTP " + error.status + ").";
  }
  return "Could not reach the chat API. Check that it is running and try again.";
}
