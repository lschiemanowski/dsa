import {
  Braces,
  ChevronDown,
  FileCheck2,
  NotebookText,
  Play,
} from "lucide-react";
import { isLosslessNumber, parse, type LosslessNumber } from "lossless-json";
import { useMemo, type Ref } from "react";
import type { AnswerSchema, Decision, Proposal, Result } from "./api";

type Scalar = string | boolean | LosslessNumber | number | null;
type Json = Scalar | Json[] | JsonObject;
interface JsonObject {
  [key: string]: Json;
}

type Schema = Omit<AnswerSchema, "properties" | "items"> & {
  properties?: Record<string, Schema>;
  items?: Schema;
  enum?: Json[];
  const?: Json;
  minimum?: Scalar;
  maximum?: Scalar;
  exclusiveMinimum?: Scalar;
  exclusiveMaximum?: Scalar;
  multipleOf?: Scalar;
  minItems?: Scalar;
  maxItems?: Scalar;
  minLength?: Scalar;
  maxLength?: Scalar;
  uniqueItems?: boolean;
};

function label(key: string, schema?: AnswerSchema) {
  const words = key.replace(/[_-]+/g, " ").trim();
  return (
    schema?.title?.trim() ||
    (words
      ? words.charAt(0).toUpperCase() + words.slice(1)
      : key.trim() || "Unnamed field")
  );
}

function jsonText(value: Json): string {
  if (isLosslessNumber(value)) return value.toString();
  if (Array.isArray(value)) return value.map(jsonText).join(", ");
  if (value !== null && typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => label(key) + " = " + jsonText(item))
      .join(", ");
  }
  return value === null ? "Null" : String(value);
}

function restrictions(schema: Schema) {
  const limits = [
    ["At least", schema.minimum],
    ["At most", schema.maximum],
    ["Greater than", schema.exclusiveMinimum],
    ["Less than", schema.exclusiveMaximum],
    ["Multiple of", schema.multipleOf],
    ["Minimum items", schema.minItems],
    ["Maximum items", schema.maxItems],
    ["Minimum characters", schema.minLength],
    ["Maximum characters", schema.maxLength],
  ] as const;
  return [
    ...limits.flatMap(([name, value]) =>
      value === undefined ? [] : [name + " " + jsonText(value)],
    ),
    ...(schema.enum
      ? ["Allowed values: " + schema.enum.map(jsonText).join(", ")]
      : []),
    ...(schema.const !== undefined
      ? ["Fixed value: " + jsonText(schema.const)]
      : []),
    ...(schema.uniqueItems ? ["Unique items"] : []),
  ];
}

function fieldType(schema: AnswerSchema) {
  switch (schema.type) {
    case "array":
      return schema.items?.type === "object" ? "Rows" : "List";
    case "object":
      return "Group of fields";
    case "string":
      return "Text";
    case "integer":
      return "Whole number";
    case "number":
      return "Number";
    case "boolean":
      return "Yes or no";
    case "null":
      return "Empty value";
    default:
      return schema.type;
  }
}

function OutputFields({ schema }: { schema: Schema }) {
  if (!schema.properties) return null;
  return (
    <ul className="output-fields">
      {Object.entries(schema.properties).map(([key, field]) => (
        <li key={key}>
          <strong>{label(key, field)}</strong>{" "}
          <span className="field-type">{fieldType(field)}</span>
          <SchemaDetails schema={field} />
        </li>
      ))}
    </ul>
  );
}

function SchemaDetails({ schema }: { schema: Schema }) {
  const limits = restrictions(schema);
  return (
    <>
      {schema.description && (
        <div className="field-note">{schema.description}</div>
      )}
      {limits.length > 0 && (
        <div className="field-note">{limits.join(" · ")}</div>
      )}
      <OutputFields schema={schema} />
      {schema.items && <SchemaDetails schema={schema.items} />}
    </>
  );
}

export function JsonDisclosure({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <details className="json-disclosure">
      <summary>
        <Braces size={15} />
        {title}
        <ChevronDown size={15} className="disclosure-arrow" />
      </summary>
      <pre>
        <code>{value}</code>
      </pre>
    </details>
  );
}

export function ProposalCard({
  proposal,
  pending,
  disabled,
  onDecision,
}: {
  proposal: Proposal;
  pending: boolean;
  disabled: boolean;
  onDecision: (choice: Decision) => void;
}) {
  const { payload } = proposal;
  const { interpretation: scope } = payload;
  const schema = useMemo(
    () =>
      (parse(proposal.request_json) as { answer_schema: Schema }).answer_schema,
    [proposal.request_json],
  );
  const fields: [string, string | string[] | null][] = [
    ["Measure", scope.measure],
    ["Population", scope.population],
    ["Time window", scope.time_window],
    ["Group by", scope.group_by],
    ["Filters", scope.filters],
    ["Units", scope.units],
  ];
  return (
    <section className="analysis-card proposal" aria-label="Analysis proposal">
      <header className="card-header">
        <FileCheck2 size={18} />
        <h2>Analysis proposal</h2>
      </header>
      <div className="card-body">
        <p className="proposal-question">{payload.question}</p>
        <dl className="scope">
          {fields
            .filter(([, value]) => value?.length)
            .map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>
                  {Array.isArray(value) ? (
                    <ul>
                      {value.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  ) : (
                    value
                  )}
                </dd>
              </div>
            ))}
        </dl>
        <div className="card-section">
          <h3>Requested outputs</h3>
          <SchemaDetails schema={schema} />
          {payload.allow_plots && (
            <p className="field-note">Plots may be included.</p>
          )}
        </div>
        {payload.analysis_guidance && (
          <div className="card-section">
            <h3>Proposed analysis guidance</h3>
            <p className="field-note">
              Based on synthetic data. The analysis model may revise this
              approach.
            </p>
            <pre className="guidance">{payload.analysis_guidance}</pre>
          </div>
        )}
        <JsonDisclosure
          title="View request JSON"
          value={proposal.request_json}
        />
      </div>
      {pending && (
        <footer className="card-actions">
          <p>Approval runs this request against the private database.</p>
          <div className="button-row">
            <button
              type="button"
              className="button primary"
              disabled={disabled}
              onClick={() => onDecision("approve")}
            >
              <Play size={15} /> Run analysis
            </button>
            <button
              type="button"
              className="button secondary"
              disabled={disabled}
              onClick={() => onDecision("revise")}
            >
              Keep refining
            </button>
          </div>
        </footer>
      )}
    </section>
  );
}

function isScalar(value: Json): value is Scalar {
  return value === null || typeof value !== "object" || isLosslessNumber(value);
}

function flatRow(value: Json): value is Record<string, Scalar> {
  return (
    !isScalar(value) &&
    !Array.isArray(value) &&
    Object.values(value).every(isScalar)
  );
}

function ScalarValue({ value }: { value: Scalar }) {
  if (value === null) return <span className="muted">Not available</span>;
  if (typeof value === "boolean") return <>{value ? "Yes" : "No"}</>;
  return (
    <span className={typeof value === "string" ? "literal" : "numeric"}>
      {String(value)}
    </span>
  );
}

function AnswerValue({
  value,
  schema,
}: {
  value: Json;
  schema?: AnswerSchema;
}) {
  if (isScalar(value)) return <ScalarValue value={value} />;
  if (!Array.isArray(value)) {
    return (
      <dl className="answer-fields">
        {Object.entries(value).map(([key, field]) => (
          <div key={key}>
            <dt>{label(key, schema?.properties?.[key])}</dt>
            <dd>
              <AnswerValue value={field} schema={schema?.properties?.[key]} />
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  if (value.length === 0) return <span className="muted">No items</span>;

  if (value.every(flatRow)) {
    const columns = Object.keys(value[0]);
    if (
      columns.length > 0 &&
      value.every(
        (row) =>
          columns.every((key) => Object.hasOwn(row, key)) &&
          Object.keys(row).length === columns.length,
      )
    ) {
      return (
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Answer table"
        >
          <table>
            <thead>
              <tr>
                {columns.map((key) => (
                  <th key={key} scope="col">
                    {label(key, schema?.items?.properties?.[key])}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {value.map((row, index) => (
                <tr key={index}>
                  {columns.map((key) => (
                    <td key={key}>
                      <ScalarValue value={row[key]} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
  }
  return (
    <ul className="answer-list">
      {value.map((item, index) => (
        <li key={index}>
          <AnswerValue value={item} schema={schema?.items} />
        </li>
      ))}
    </ul>
  );
}

export function ResultCard({
  result,
  schema,
  notebookOpen,
  toggleNotebook,
  notebookButton,
}: {
  result: Result;
  schema?: AnswerSchema;
  notebookOpen: boolean;
  toggleNotebook: () => void;
  notebookButton: Ref<HTMLButtonElement>;
}) {
  const answer = useMemo(
    () =>
      result.response_json === null
        ? null
        : (parse(result.response_json) as Json),
    [result.response_json],
  );
  return (
    <section className="analysis-card result" aria-label="Analysis result">
      <header className="card-header">
        <FileCheck2 size={18} />
        <h2>Analysis result</h2>
      </header>
      <div className="card-body">
        <AnswerValue value={answer} schema={schema} />
        {result.response_json !== null && (
          <JsonDisclosure
            title="View response JSON"
            value={result.response_json}
          />
        )}
      </div>
      <footer className="card-actions">
        {result.notebook ? (
          <button
            type="button"
            className="button secondary"
            ref={notebookButton}
            aria-expanded={notebookOpen}
            aria-controls={notebookOpen ? "notebook-pane" : undefined}
            onClick={toggleNotebook}
          >
            <NotebookText size={16} />{" "}
            {notebookOpen ? "Hide notebook" : "Open notebook"}
          </button>
        ) : (
          <p>No verified notebook was produced.</p>
        )}
      </footer>
    </section>
  );
}
