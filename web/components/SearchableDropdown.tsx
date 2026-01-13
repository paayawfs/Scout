"use client";

import { useState, useRef, useEffect } from "react";

interface SearchableDropdownProps {
    options: string[];
    value: string | null;
    onChange: (value: string | null) => void;
    placeholder?: string;
}

export default function SearchableDropdown({
    options,
    value,
    onChange,
    placeholder = "Select...",
}: SearchableDropdownProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const wrapperRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    useEffect(() => {
        if (isOpen && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isOpen]);

    const filteredOptions = options.filter((option) =>
        option.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="relative w-full" ref={wrapperRef}>
            {/* Trigger Button */}
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                className={`w-full text-left px-4 py-3 rounded-lg border-2 flex items-center justify-between transition-colors ${isOpen ? "border-accent ring-0" : "border-black hover:border-accent"
                    } bg-white text-black shadow-sm`}
            >
                <span className={`block truncate ${!value ? "text-gray-500" : "font-bold"}`}>
                    {value || placeholder}
                </span>
                <svg
                    className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? "rotate-180" : ""}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={3}
                        d="M19 9l-7 7-7-7"
                    />
                </svg>
            </button>

            {/* Dropdown Menu */}
            {isOpen && (
                <div className="absolute z-[9999] w-full mt-2 bg-white border-2 border-black rounded-xl shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] overflow-hidden animate-fadeIn">
                    {/* Search Input */}
                    <div className="p-2 border-b-2 border-black bg-gray-50">
                        <input
                            ref={inputRef}
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Type to search..."
                            className="w-full px-3 py-2 bg-white border-2 border-black rounded-lg text-sm text-black placeholder-gray-500 focus:outline-none focus:border-accent"
                        />
                    </div>

                    {/* Options List */}
                    <div className="max-h-60 overflow-y-auto">
                        <button
                            type="button"
                            onClick={() => {
                                onChange(null);
                                setIsOpen(false);
                                setSearchQuery("");
                            }}
                            className={`w-full text-left px-4 py-2 text-sm hover:bg-gray-100 transition-colors border-b border-gray-100 ${!value ? "text-accent font-bold bg-blue-50" : "text-gray-700"
                                }`}
                        >
                            All Nations
                        </button>
                        {filteredOptions.length > 0 ? (
                            filteredOptions.map((option) => (
                                <button
                                    key={option}
                                    type="button"
                                    onClick={() => {
                                        onChange(option);
                                        setIsOpen(false);
                                        setSearchQuery("");
                                    }}
                                    className={`w-full text-left px-4 py-2 text-sm hover:bg-gray-100 transition-colors border-b border-gray-100 ${value === option
                                            ? "text-accent font-bold bg-blue-50"
                                            : "text-black"
                                        }`}
                                >
                                    {option}
                                </button>
                            ))
                        ) : (
                            <div className="px-4 py-3 text-sm text-gray-500 text-center font-medium">
                                No results found
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
