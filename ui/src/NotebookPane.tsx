import { Download, NotebookText, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { readNotebook, type Notebook } from "./api";
import { SafeMarkdown } from "./SafeMarkdown";

type ObjectValue = Record<string, unknown>;

function object(value: unknown): ObjectValue | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as ObjectValue)
    : undefined;
}

function text(value: unknown): string {
  if (typeof value === "string") return value;
  return Array.isArray(value) && value.every((line) => typeof line === "string")
    ? value.join("")
    : "";
}

function Output({
  value,
  plot,
}: {
  value: unknown;
  plot: ObjectValue | undefined;
}) {
  const output = object(value);
  if (!output) return null;
  const data = object(output.data);
  const png = text(data?.["image/png"]);
  if (/^iVBORw0KGgo[A-Za-z0-9+/]*={0,2}$/.test(png)) {
    const source = "data:image/png;base64," + png;
    const title = text(plot?.title) || "Notebook plot";
    return (
      <figure className="plot">
        <img src={source} alt={title} loading="lazy" />
        <figcaption>
          <span>{title}</span>
          <a
            className="button secondary compact"
            href={source}
            download={text(plot?.filename) || "plot.png"}
          >
            <Download size={14} /> Download plot
          </a>
        </figcaption>
      </figure>
    );
  }
  const plain = text(output.text) || text(data?.["text/plain"]);
  const trace = Array.isArray(output.traceback)
    ? output.traceback.map(text).join("\n")
    : "";
  return plain || trace ? (
    <pre className="cell-output">{plain || trace}</pre>
  ) : null;
}

function NotebookCell({ cell }: { cell: ObjectValue }) {
  const source = text(cell.source);
  if (cell.cell_type === "markdown")
    return <SafeMarkdown>{source}</SafeMarkdown>;
  const outputs: unknown[] = Array.isArray(cell.outputs) ? cell.outputs : [];
  const plot = object(object(cell.metadata)?.dsa_plot);
  return (
    <div className="notebook-cell">
      <pre className="cell-source">
        <code>{source}</code>
      </pre>
      {outputs.map((output, index) => (
        <Output key={index} value={output} plot={plot} />
      ))}
    </div>
  );
}

export function NotebookPane({
  conversationId,
  notebook,
  close,
}: {
  conversationId: string;
  notebook: Notebook;
  close: () => void;
}) {
  const [cells, setCells] = useState<ObjectValue[] | null>(null);
  const [failed, setFailed] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.focus();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void readNotebook(conversationId, controller.signal)
      .then((value) => {
        const notebook = object(value);
        if (notebook?.nbformat !== 4 || !Array.isArray(notebook.cells)) {
          throw new Error("Invalid notebook");
        }
        const cells: unknown[] = notebook.cells;
        if (!controller.signal.aborted)
          setCells(cells.map(object).filter((cell) => cell !== undefined));
      })
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => controller.abort();
  }, [conversationId]);

  return (
    <aside
      id="notebook-pane"
      className="notebook"
      aria-labelledby="notebook-heading"
      onKeyDown={(event) => {
        if (event.key === "Escape") close();
      }}
    >
      <header className="notebook-header">
        <div className="notebook-title">
          <NotebookText size={19} />
          <h2 id="notebook-heading" ref={heading} tabIndex={-1}>
            Derivation notebook
          </h2>
        </div>
        <div className="notebook-actions">
          <a
            className="button secondary compact"
            href={notebook.url}
            download="dsa-derivation.ipynb"
          >
            <Download size={15} /> Download
          </a>
          <button
            type="button"
            className="button icon-button"
            aria-label="Close notebook"
            onClick={close}
          >
            <X size={20} />
          </button>
        </div>
      </header>
      <div className="notebook-content">
        {failed ? (
          <p role="alert">The notebook preview could not be loaded.</p>
        ) : cells === null ? (
          <p role="status">Loading notebook…</p>
        ) : cells.length === 0 ? (
          <p>This notebook has no cells.</p>
        ) : (
          cells.map((cell, index) => <NotebookCell key={index} cell={cell} />)
        )}
      </div>
    </aside>
  );
}
