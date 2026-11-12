import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  createConversation,
  decide,
  explainError,
  readConversation,
  sendMessage,
  type ChatStatus,
  type Conversation,
  type Decision,
  type Proposal,
} from "./api";

const sessionKey = "dsa.active-conversation";

function storedConversation(): string | null {
  try {
    return sessionStorage.getItem(sessionKey);
  } catch {
    return null;
  }
}

function rememberConversation(id: string | null) {
  try {
    if (id === null) sessionStorage.removeItem(sessionKey);
    else sessionStorage.setItem(sessionKey, id);
  } catch {
    // The chat can still run when browser storage is unavailable.
  }
}

function wasRemoved(error: unknown) {
  return error instanceof ApiError && error.code === "conversation_not_found";
}

export function isWorking(status?: ChatStatus) {
  return status === "clarifying" || status === "running";
}

export function useConversation() {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [busy, setBusy] = useState(true);
  const [pendingText, setPendingText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expired, setExpired] = useState(false);
  const [restoreAttempt, setRestoreAttempt] = useState(0);
  const inFlight = useRef(true);

  useEffect(() => {
    const controller = new AbortController();
    const signal = controller.signal;
    async function restore() {
      const stored = storedConversation();
      try {
        let state = null;
        if (stored) {
          try {
            state = await readConversation(stored, signal);
          } catch (error) {
            if (!wasRemoved(error)) throw error;
            rememberConversation(null);
          }
        }
        state ??= await createConversation(signal);
        if (!signal.aborted) {
          rememberConversation(state.id);
          setConversation(state);
        }
      } catch (error) {
        if (!signal.aborted) setError(explainError(error));
      } finally {
        if (!signal.aborted) {
          inFlight.current = false;
          setBusy(false);
        }
      }
    }
    void restore();
    return () => controller.abort();
  }, [restoreAttempt]);

  const id = conversation?.id;
  const status = conversation?.status;
  useEffect(() => {
    if (!id || !isWorking(status) || expired) return;
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll(conversationId: string) {
      try {
        const state = await readConversation(conversationId, controller.signal);
        if (controller.signal.aborted) return;
        setConversation(state);
        setError(null);
        if (!isWorking(state.status)) return;
      } catch (error) {
        if (controller.signal.aborted) return;
        setError(explainError(error));
        if (wasRemoved(error)) {
          rememberConversation(null);
          setExpired(true);
          return;
        }
      }
      timer = window.setTimeout(() => void poll(conversationId), 1250);
    }
    timer = window.setTimeout(() => void poll(id), 800);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [id, status, expired]);

  function retryRestore() {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError(null);
    setRestoreAttempt((attempt) => attempt + 1);
  }

  async function update(
    action: () => Promise<Conversation>,
    options?: { text?: string; accepted: (state: Conversation) => boolean },
  ) {
    if (inFlight.current) return false;
    inFlight.current = true;
    setBusy(true);
    setPendingText(options?.text ?? null);
    setError(null);
    try {
      const state = await action();
      rememberConversation(state.id);
      setConversation(state);
      setExpired(false);
      return true;
    } catch (error) {
      let problem = error;
      if (!wasRemoved(error) && id && options) {
        // A lost response may still have been accepted by the server.
        try {
          const state = await readConversation(id);
          setConversation(state);
          const ambiguous =
            !(error instanceof ApiError) ||
            error.status >= 500 ||
            error.code === "invalid_response";
          if (ambiguous && options.accepted(state)) return true;
        } catch (refreshError) {
          if (wasRemoved(refreshError)) problem = refreshError;
        }
      }
      if (wasRemoved(problem)) {
        rememberConversation(null);
        setExpired(true);
      }
      setError(explainError(problem));
      return false;
    } finally {
      inFlight.current = false;
      setPendingText(null);
      setBusy(false);
    }
  }

  return {
    conversation,
    busy,
    pendingText,
    error,
    expired,
    retryRestore,
    start: () => update(() => createConversation()),
    send: (text: string) =>
      id
        ? update(() => sendMessage(id, text), {
            text,
            accepted: (state) => {
              const message =
                state.messages[conversation?.messages.length ?? 0];
              return message?.role === "user" && message.text === text;
            },
          })
        : Promise.resolve(false),
    decide: (proposal: Proposal, choice: Decision) =>
      id
        ? update(() => decide(id, proposal, choice), {
            accepted: (state) => {
              if (choice === "revise") {
                return (
                  state.status === "open" && state.pending_proposal_id === null
                );
              }
              return state.result
                ? state.result.proposal_id === proposal.id
                : (state.status === "running" || state.status === "failed") &&
                    state.messages.filter((message) => message.proposal).at(-1)
                      ?.proposal?.id === proposal.id;
            },
          })
        : Promise.resolve(false),
  };
}
