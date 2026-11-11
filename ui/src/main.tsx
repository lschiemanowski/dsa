import { createRoot } from "react-dom/client";
import mark from "../../apps/private_data_chat/public/dsa-mark.svg";
import { App } from "./App";
import "./styles.css";

document.querySelector<HTMLLinkElement>('link[rel="icon"]')!.href = mark;

createRoot(document.getElementById("root")!).render(<App />);
