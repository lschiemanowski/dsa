import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { ArrowUp, LoaderCircle, Plus } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import mark from "../../apps/private_data_chat/public/dsa-mark.svg";
import { ProposalCard, ResultCard } from "./AnalysisCards";
import type { Conversation } from "./api";
import { NotebookPane } from "./NotebookPane";
import { SafeMarkdown } from "./SafeMarkdown";
import { isWorking, useConversation } from "./use-conversation";

type Entry = Conversation["messages"][number] & { id: string };

function convertMessage(entry: Entry): ThreadMessageLike {
  return {
    id: entry.id,
    role: entry.role,
    content: [{ type: "text", text: entry.text }],
  };
}

function Chat({
  session,
  notebookOpen,
  toggleNotebook,
  notebookButton,
}: {
  session: ReturnType<typeof useConversation>;
  notebookOpen: boolean;
  toggleNotebook: () => void;
  notebookButton: React.Ref<HTMLButtonElement>;
}) {
  const { conversation, busy, pendingText, error, expired } = session;
  const status = conversation?.status;
  const result = status === "succeeded" ? conversation?.result : null;
  const terminal = status === "succeeded" || status === "failed";
  const resultId = result?.run_id;
  const messages = useMemo<Entry[]>(() => {
    const entries = (conversation?.messages ?? []).map((message, index) => ({
      ...message,
      id: conversation?.id + "-" + index,
    }));
    if (pendingText !== null)
      entries.push({
        id: conversation?.id + "-" + entries.length,
        role: "user",
        text: pendingText,
        proposal: null,
      });
    if (resultId)
      entries.push({
        id: resultId,
        role: "assistant",
        text: "Analysis result",
        proposal: null,
      });
    return entries;
  }, [conversation, pendingText, resultId]);
  const byId = new Map(messages.map((entry) => [entry.id, entry]));
  const disabled = expired || status !== "open" || busy;
  const runtime = useExternalStoreRuntime({
    messages,
    convertMessage,
    isRunning: busy || isWorking(status),
    isSendDisabled: disabled,
    onNew: async (message) => {
      const text = message.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("\n")
        .trim();
      if (text && !(await session.send(text)))
        runtime.thread.composer.setText(text);
    },
  });
  const statusText = expired
    ? ""
    : status === "running"
      ? "Running the approved analysis…"
      : status === "clarifying" || pendingText !== null
        ? "Preparing a clarification…"
        : status === "awaiting_decision"
          ? "Review the proposal above to continue."
          : terminal
            ? "Start a new conversation to ask another question."
            : "";

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="chat">
        <ThreadPrimitive.Viewport className="chat-viewport">
          <div className="transcript">
            <ThreadPrimitive.Empty>
              {conversation?.welcome && (
                <div className="welcome">
                  <SafeMarkdown>{conversation.welcome}</SafeMarkdown>
                </div>
              )}
              {!conversation && !error && (
                <p role="status" className="muted">
                  Connecting to the chat API…
                </p>
              )}
            </ThreadPrimitive.Empty>
            <ThreadPrimitive.Messages>
              {({ message }) => {
                const entry = byId.get(message.id);
                if (!entry) return null;
                const proposal = entry.proposal;
                if (result && message.id === resultId) {
                  const schema = conversation?.messages.find(
                    ({ proposal }) => proposal?.id === result.proposal_id,
                  )?.proposal?.payload.answer_schema;
                  return (
                    <MessagePrimitive.Root className="message assistant">
                      <ResultCard
                        result={result}
                        schema={schema}
                        notebookOpen={notebookOpen}
                        toggleNotebook={toggleNotebook}
                        notebookButton={notebookButton}
                      />
                    </MessagePrimitive.Root>
                  );
                }
                return (
                  <MessagePrimitive.Root className={"message " + entry.role}>
                    {proposal ? (
                      <ProposalCard
                        proposal={proposal}
                        pending={
                          conversation?.pending_proposal_id === proposal.id
                        }
                        disabled={busy || expired}
                        onDecision={(decision) =>
                          void session.decide(proposal, decision)
                        }
                      />
                    ) : entry.role === "user" ? (
                      <p className="user-bubble">{entry.text}</p>
                    ) : (
                      <SafeMarkdown>{entry.text}</SafeMarkdown>
                    )}
                  </MessagePrimitive.Root>
                );
              }}
            </ThreadPrimitive.Messages>
            {status === "failed" && (
              <div className="run-error" role="alert">
                <strong>The analysis ended without an answer.</strong>
                {conversation?.failure_code && (
                  <code>{conversation.failure_code}</code>
                )}
              </div>
            )}
          </div>
        </ThreadPrimitive.Viewport>
        <div className="composer-area">
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {statusText && (
            <div className="status" role="status">
              {(isWorking(status) || pendingText !== null) && (
                <LoaderCircle size={16} className="spinner" />
              )}
              {statusText}
            </div>
          )}
          {conversation && !terminal && !expired && (
            <ComposerPrimitive.Root className="composer">
              <ComposerPrimitive.Input
                aria-label="Ask a question"
                placeholder="Ask a quantitative question…"
                disabled={disabled}
                maxLength={8192}
                rows={2}
              />
              <ComposerPrimitive.Send
                className="button primary send-button"
                aria-label="Send question"
              >
                <ArrowUp size={19} />
              </ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
          )}
        </div>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

export function App() {
  const session = useConversation();
  const [notebookOpen, setNotebookOpen] = useState(false);
  const notebookButton = useRef<HTMLButtonElement>(null);
  const { conversation, expired, busy } = session;
  const notebook = conversation?.result?.notebook;
  const showNotebook = Boolean(notebook && notebookOpen && !expired);

  function closeNotebook() {
    setNotebookOpen(false);
    window.requestAnimationFrame(() => notebookButton.current?.focus());
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src={mark} alt="" />
          <h1>Private Data Chat</h1>
        </div>
        <button
          type="button"
          className="button secondary new-conversation"
          disabled={busy || (!expired && isWorking(conversation?.status))}
          onClick={() =>
            void session.start().then((started) => {
              if (started) setNotebookOpen(false);
            })
          }
        >
          <Plus size={17} /> New conversation
        </button>
      </header>
      <main className="workspace" data-notebook-open={showNotebook}>
        <Chat
          key={conversation?.id}
          session={session}
          notebookOpen={showNotebook}
          toggleNotebook={() =>
            showNotebook ? closeNotebook() : setNotebookOpen(true)
          }
          notebookButton={notebookButton}
        />
        {showNotebook && notebook && conversation && (
          <NotebookPane
            conversationId={conversation.id}
            notebook={notebook}
            close={closeNotebook}
          />
        )}
      </main>
    </div>
  );
}
