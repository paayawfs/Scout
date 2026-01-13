"use client";

import { useState } from "react";
import SearchableDropdown from "./SearchableDropdown";

interface FilterBarProps {
    positions: string[];
    nations: string[];
    onFilterChange: (filters: {
        position: string | null;
        nation: string | null;
        maxAge: number | null;
    }) => void;
}

export default function FilterBar({ positions, nations, onFilterChange }: FilterBarProps) {
    const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
    const [selectedNation, setSelectedNation] = useState<string | null>(null);
    const [maxAge, setMaxAge] = useState<string>("");

    const handlePositionClick = (pos: string) => {
        const newPos = selectedPosition === pos ? null : pos;
        setSelectedPosition(newPos);
        onFilterChange({
            position: newPos,
            nation: selectedNation,
            maxAge: maxAge ? parseInt(maxAge) : null,
        });
    };



    const handleAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const age = e.target.value;
        setMaxAge(age);
        onFilterChange({
            position: selectedPosition,
            nation: selectedNation,
            maxAge: age ? parseInt(age) : null,
        });
    };

    const clearFilters = () => {
        setSelectedPosition(null);
        setSelectedNation(null);
        setMaxAge("");
        onFilterChange({ position: null, nation: null, maxAge: null });
    };

    const hasFilters = selectedPosition || selectedNation || maxAge;

    return (
        <div className="space-y-4">
            {/* Position filters */}
            <div>
                <label className="text-sm text-gray-500 font-medium uppercase tracking-wide mb-2 block">
                    Position
                </label>
                <div className="flex flex-wrap gap-2">
                    {positions.map((pos) => (
                        <button
                            key={pos}
                            onClick={() => handlePositionClick(pos)}
                            className={`filter-chip ${selectedPosition === pos ? "active" : ""}`}
                        >
                            {pos}
                        </button>
                    ))}
                </div>
            </div>

            {/* Nation and Age filters */}
            <div className="flex flex-wrap gap-4">
                <div className="flex-1 min-w-[200px]">
                    <label className="text-sm text-gray-500 font-medium uppercase tracking-wide mb-2 block">
                        Nationality
                    </label>
                    <SearchableDropdown
                        options={nations}
                        value={selectedNation}
                        onChange={(nation) => {
                            setSelectedNation(nation);
                            onFilterChange({
                                position: selectedPosition,
                                nation,
                                maxAge: maxAge ? parseInt(maxAge) : null,
                            });
                        }}
                        placeholder="All Nations"
                    />
                </div>

                <div className="w-32">
                    <label className="text-sm text-gray-500 font-medium uppercase tracking-wide mb-2 block">
                        Max Age
                    </label>
                    <input
                        type="number"
                        value={maxAge}
                        onChange={handleAgeChange}
                        placeholder="e.g. 25"
                        min="15"
                        max="45"
                        className="w-full"
                    />
                </div>
            </div>

            {/* Clear filters */}
            {hasFilters && (
                <button
                    onClick={clearFilters}
                    className="text-sm text-primary hover:text-accent transition-colors font-medium flex items-center gap-1"
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    Clear all filters
                </button>
            )}
        </div>
    );
}
