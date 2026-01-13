"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";

interface Toast {
    id: string;
    message: string;
    type: "error" | "success" | "info";
}

interface ToastContextType {
    toasts: Toast[];
    showToast: (message: string, type?: Toast["type"]) => void;
    removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error("useToast must be used within a ToastProvider");
    }
    return context;
}

export function ToastProvider({ children }: { children: ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const showToast = useCallback((message: string, type: Toast["type"] = "error") => {
        const id = Math.random().toString(36).substring(7);
        setToasts((prev) => [...prev, { id, message, type }]);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            setToasts((prev) => prev.filter((t) => t.id !== id));
        }, 5000);
    }, []);

    const removeToast = useCallback((id: string) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    }, []);

    return (
        <ToastContext.Provider value={{ toasts, showToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} removeToast={removeToast} />
        </ToastContext.Provider>
    );
}

function ToastContainer({ toasts, removeToast }: { toasts: Toast[]; removeToast: (id: string) => void }) {
    if (toasts.length === 0) return null;

    return (
        <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
            {toasts.map((toast) => (
                <div
                    key={toast.id}
                    className={`
                        flex items-center gap-3 px-4 py-3 min-w-[300px] max-w-[400px] 
                        border-2 border-black rounded-lg shadow-[4px_4px_0px_0px_#000000]
                        animate-fadeIn
                        ${toast.type === "error" ? "bg-red-50 text-red-800" : ""}
                        ${toast.type === "success" ? "bg-green-50 text-green-800" : ""}
                        ${toast.type === "info" ? "bg-blue-50 text-blue-800" : ""}
                    `}
                >
                    {/* Icon */}
                    {toast.type === "error" && (
                        <svg className="w-5 h-5 flex-shrink-0 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    )}
                    {toast.type === "success" && (
                        <svg className="w-5 h-5 flex-shrink-0 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                    )}
                    {toast.type === "info" && (
                        <svg className="w-5 h-5 flex-shrink-0 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    )}

                    {/* Message */}
                    <p className="font-medium text-sm flex-1">{toast.message}</p>

                    {/* Close button */}
                    <button
                        onClick={() => removeToast(toast.id)}
                        className="flex-shrink-0 hover:opacity-70 transition-opacity"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
            ))}
        </div>
    );
}
