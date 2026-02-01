"use client";

import { useState } from "react";

type Source = {
  doc: string;
  version: string;
  section: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const EXAMPLE_QUESTIONS = [
  "Why did the SEC introduce climate-related disclosure requirements?",
  "What changed between the 2022 proposed rule and the 2024 final rule?",
  "Are companies required to disclose Scope 3 emissions?",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [version, setVersion] = useState<string | null>(null);

  async function sendMessage(query: string) {
    if (!query.trim()) return;

    setMessages((prev) => [...prev, { role: "user", content: query }]);

    setLoading(true);
    setInput("");

    try {
      const res = await fetch(
        "https://regulens-2q8t.onrender.com/disclosure-analysis",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            version,
            mode: "fast",
          }),
        },
      );

      const data = await res.json();
      console.log(data);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "An error occurred while retrieving the answer. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#0F0F0F] text-gray-100">
      <div className="max-w-4xl my-12 mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-3 text-white bg-clip-text text-center">
            ReguLens
          </h1>
          <p className="text-gray-300 text-xl text-center">
            Regulatory Q&A grounded in official SEC climate disclosure rules
          </p>
        </div>

        {/* Controls */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Document Version
          </label>
          <select
            className="bg-[#1A1A1A] border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer hover:bg-[#222222] transition-colors w-full sm:w-auto"
            value={version ?? ""}
            onChange={(e) => setVersion(e.target.value || null)}
          >
            <option value="">All versions (prefer final rule)</option>
            <option value="2024_final">2024 Final Rule</option>
            <option value="2022_proposed">2022 Proposed Rule</option>
          </select>
        </div>

        {/* Example Questions */}
        {messages.length === 0 && (
          <div className="mb-8 bg-[#1A1A1A] rounded-xl p-6 border border-gray-800">
            <p className="text-sm font-medium text-gray-300 mb-3">
              Try asking:
            </p>
            <div className="flex flex-col gap-2">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-sm text-left bg-[#262626] hover:bg-[#2E2E2E] border border-gray-700 rounded-lg px-4 py-3 text-gray-200 transition-all hover:border-blue-500/50 cursor-pointer"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat Messages */}
        <div className="space-y-4 mb-6 min-h-50">
          {messages.map((msg, i) => (
            <div key={i} className="animate-fadeIn">
              <div
                className={`rounded-xl p-4 ${
                  msg.role === "user"
                    ? "bg-blue-600/15 border border-blue-500/20 ml-8"
                    : "bg-[#1A1A1A] border border-gray-800 mr-8"
                }`}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-linear-to-br from-cyan-500 to-blue-500 text-white"
                    }`}
                  >
                    {msg.role === "user" ? "You" : "AI"}
                  </div>
                  <p className="whitespace-pre-wrap text-gray-200 leading-relaxed flex-1 pt-1">
                    {msg.content}
                  </p>
                </div>
              </div>

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 ml-11 mr-8 bg-[#141414] rounded-lg p-4 border border-gray-800">
                  <p className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-2">
                    Sources
                  </p>
                  <ul className="space-y-2">
                    {msg.sources.map((s, idx) => (
                      <li
                        key={idx}
                        className="text-sm text-gray-400 flex items-start gap-2"
                      >
                        <span className="text-cyan-400 shrink-0">→</span>
                        <span>
                          <span className="text-gray-300 font-medium">
                            {s.doc}
                          </span>{" "}
                          <span className="text-gray-500">({s.version})</span>{" "}
                          <span className="text-gray-400">
                            — Section {s.section}
                          </span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-3 text-gray-400 ml-11">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
              </div>
              <p className="text-sm">Analyzing regulatory context…</p>
            </div>
          )}
        </div>

        {/* Input Form */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage(input);
          }}
          className="sticky bottom-4 bg-[#1A1A1A]/98 backdrop-blur-sm border border-gray-700 rounded-xl p-2 shadow-2xl"
        >
          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about SEC climate disclosure rules…"
              className="flex-1 bg-transparent px-4 py-3 text-gray-200 placeholder-gray-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-white text-black font-semibold px-5 py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all cursor-pointer shadow-lg hover:shadow-blue-500/25"
            >
              {loading ? "Sending..." : "Ask"}
            </button>
          </div>
        </form>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </main>
  );
}
