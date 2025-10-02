import { useState, useRef, useEffect } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [documentText, setDocumentText] = useState("");
  const [uploadedFilename, setUploadedFilename] = useState("");
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat, loading]);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setUploadedFilename(data.filename);
    setLoading(false);
    alert("File uploaded!");
    // Optionally fetch document preview here
    setDocumentText(""); // Clear previous preview
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question) return;
    setChat((prev) => [...prev, { question, answer: null }]);
    setLoading(true);
    setQuestion("");
    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setChat((prev) =>
      prev.map((msg, idx) =>
        idx === prev.length - 1 ? { ...msg, answer: data.answer } : msg
      )
    );
    setLoading(false);
  };

  const handleSummarize = async () => {
    if (!uploadedFilename) return;
    setSummary("Loading...");
    const res = await fetch("http://localhost:8000/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: uploadedFilename, style: "brief" }),
    });
    const data = await res.json();
    setSummary(data.summary);
  };

  // Optionally, fetch document preview after upload
  // For now, just show a placeholder or the extracted text if available

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Left: Chat */}
      <div className="w-1/2 flex flex-col bg-white shadow-lg">
        <header className="bg-blue-600 text-white p-4 text-xl font-bold shadow">
          AI Chat
        </header>
        {/* File upload */}
        <div className="flex items-center gap-2 p-4 border-b">
          <input
            type="file"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
          <button
            onClick={handleUpload}
            disabled={loading || !file}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            Upload
          </button>
        </div>
        {/* Chat area */}
        <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2 bg-gray-50">
          {chat.map((c, i) => (
            <div key={i}>
              {/* User message */}
              <div className="flex justify-end mb-1">
                <div className="bg-blue-500 text-white px-4 py-2 rounded-2xl rounded-br-none max-w-xs shadow">
                  {c.question}
                </div>
              </div>
              {/* AI answer */}
              {c.answer !== null && (
                <div className="flex justify-start">
                  <div className="bg-gray-200 text-gray-800 px-4 py-2 rounded-2xl rounded-bl-none max-w-xs shadow">
                    {c.answer}
                  </div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 text-gray-500 px-4 py-2 rounded-2xl rounded-bl-none max-w-xs shadow animate-pulse">
                AI is typing...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
        {/* Input bar */}
        <form
          onSubmit={handleAsk}
          className="flex items-center gap-2 p-4 border-t bg-white"
        >
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-200"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !question}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </div>

      {/* Right: Summarization (top) + Document Preview (bottom) */}
      <div className="w-1/2 flex flex-col">
        {/* Summarization */}
        <div className="flex-1 bg-white m-2 rounded shadow p-4 flex flex-col">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-bold">Summarization</h2>
            <button
              onClick={handleSummarize}
              disabled={!uploadedFilename}
              className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 disabled:opacity-50"
            >
              Summarize
            </button>
          </div>
          <div className="overflow-y-auto flex-1">
            {summary ? (
              <p className="text-gray-800 whitespace-pre-line">{summary}</p>
            ) : (
              <p className="text-gray-400">No summary yet.</p>
            )}
          </div>
        </div>
        {/* Document Preview */}
        <div className="flex-1 bg-white m-2 rounded shadow p-4 flex flex-col">
          <h2 className="text-lg font-bold mb-2">Document Preview</h2>
          <div className="overflow-y-auto flex-1">
            {documentText ? (
              <pre className="text-gray-700 whitespace-pre-wrap">
                {documentText}
              </pre>
            ) : (
              <p className="text-gray-400">No document loaded.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
