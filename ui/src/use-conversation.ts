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
  }, []);

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

  async function update(
    action: () => Promise<Conversation>,
    text?: string,
    refresh = true,
  ) {
    if (inFlight.current) return false;
    inFlight.current = true;
    setBusy(true);
    setPendingText(text ?? null);
    setError(null);
    try {
      const state = await action();
      rememberConversation(state.id);
      setConversation(state);
      setExpired(false);
      return true;
    } catch (error) {
      setError(explainError(error));
      if (wasRemoved(error)) {
        rememberConversation(null);
        setExpired(true);
      } else if (id && refresh) {
        // A lost response may still have been accepted by the server.
        try {
          const state = await readConversation(id);
          setConversation(state);
          const firstNew = state.messages[conversation?.messages.length ?? 0];
          if (
            text !== undefined &&
            firstNew?.role === "user" &&
            firstNew.text === text
          ) {
            setError(null);
            return true;
          }
        } catch (refreshError) {
          if (wasRemoved(refreshError)) {
            rememberConversation(null);
            setExpired(true);
            setError(explainError(refreshError));
          }
        }
      }
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
    start: () => update(() => createConversation(), undefined, false),
    send: (text: string) =>
      id ? update(() => sendMessage(id, text), text) : Promise.resolve(false),
    decide: (proposal: Proposal, choice: Decision) =>
      id ? update(() => decide(id, proposal, choice)) : Promise.resolve(false),
  };
}
