"use client";

import {
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ResponsiveContainer,
    Legend,
    Tooltip,
} from "recharts";
import { METRIC_SETS, getPositionSet } from "@/lib/metrics";

interface RadarComparisonProps {
    player1: {
        name: string;
        values: number[];
        position?: string;
    };
    player2: {
        name: string;
        values: number[];
        position?: string;
    };
    labels: string[];
}

export default function RadarComparison({
    player1,
    player2,
    labels,
}: RadarComparisonProps) {
    const metrics = getPositionSet(player1.position || '');

    // Normalize values to 0-100 scale based on the actual data range
    const normalizeValue = (value: number, maxVal: number) => {
        if (maxVal === 0) return 0;
        const normalized = (value / maxVal) * 100;
        return Math.max(0, Math.min(100, normalized));
    };

    // Construct data map for easy lookup
    const p1Map = new Map();
    const p2Map = new Map();
    labels.forEach((label, i) => {
        p1Map.set(label, player1.values[i] || 0);
        p2Map.set(label, player2.values[i] || 0);
    });

    // Build ordered data for the 6 axes using dynamic METRICS
    const data = metrics.map((metric) => {
        const val1 = p1Map.get(metric.key) || 0;
        const val2 = p2Map.get(metric.key) || 0;
        // Global max scaling (relative to comparison)
        const maxVal = Math.max(val1, val2) * 1.2 || 1;

        return {
            stat: metric.label,
            value1: normalizeValue(val1, maxVal),
            value2: normalizeValue(val2, maxVal),
            raw1: val1.toFixed(2),
            raw2: val2.toFixed(2),
            fullMark: 100, // Forcing outer frame
        };
    });

    // Custom Tick with Dark Pill styling
    const CustomTick = ({ payload, x, y, cx, cy, ...rest }: any) => {
        return (
            <g transform={`translate(${x},${y})`}>
                <rect
                    x={-40}
                    y={-12}
                    width={80}
                    height={24}
                    rx={12}
                    fill="#1F1F3D"
                    fillOpacity={0.9}
                />
                <text
                    x={0}
                    y={0}
                    dy={4}
                    textAnchor="middle"
                    fill="#FFFFFF"
                    fontSize={11}
                    fontWeight={600}
                    fontFamily="Inter, sans-serif"
                >
                    {payload.value}
                </text>
            </g>
        );
    };

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-[#1F1F3D] border border-gray-600 p-3 rounded-lg shadow-xl z-50">
                    <p className="font-bold text-white uppercase mb-2 border-b border-gray-600 pb-1">{label}</p>
                    {payload.map((entry: any, index: number) => {
                        // entry.dataKey is 'value1' or 'value2'
                        const isPlayer1 = entry.dataKey === 'value1';
                        const rawVal = isPlayer1 ? entry.payload.raw1 : entry.payload.raw2;

                        return (
                            <p key={index} className="text-sm font-mono font-bold" style={{ color: entry.stroke }}>
                                {entry.name}: {rawVal}
                            </p>
                        );
                    })}
                </div>
            );
        }
        return null;
    };

    return (
        <div className="w-full h-[500px] flex items-center justify-center bg-gray-50 rounded-xl border-2 border-black">
            {/* Hexagon Chart Container */}
            <div className="w-[450px] h-[450px] relative">
                <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>

                        {/* 
                            Frame Setup:
                            - gridType="polygon" creates the hexagon.
                            - radialLines={true} draws the 6 axes.
                            - polarLines={true} draws the rings.
                            - stroke="#374151" (Medium Gray Frame).
                        */}
                        <PolarGrid
                            gridType="polygon"
                            radialLines={true}
                            stroke="#374151"
                            strokeWidth={1}
                        />

                        {/* Labels with Custom Pill */}
                        <PolarAngleAxis
                            dataKey="stat"
                            tick={<CustomTick />}
                        />

                        {/* Hidden Axis Scale */}
                        <PolarRadiusAxis
                            angle={30}
                            domain={[0, 100]}
                            tick={false}
                            axisLine={false}
                            tickCount={2} // Center + Outer Frame only
                        />

                        {/* Player 1: Magenta Filled Polygon */}
                        <Radar
                            name={player1.name}
                            dataKey="value1"
                            stroke="#EC4899"
                            strokeWidth={3}
                            fill="#EC4899"
                            fillOpacity={0.2}
                            dot={{ r: 6, fill: "#EC4899", stroke: "#FFFFFF", strokeWidth: 2 }}
                            isAnimationActive={false}
                        />

                        {/* Player 2: Purple Dashed Outline */}
                        <Radar
                            name={player2.name}
                            dataKey="value2"
                            stroke="#5B21B6"
                            strokeWidth={2}
                            fill="none" // No fill/low fill requested
                            strokeDasharray="4 4"
                            dot={{ r: 4, fill: "#5B21B6", strokeWidth: 0 }}
                            isAnimationActive={false}
                        />

                        <Tooltip content={<CustomTooltip />} />

                        <Legend
                            wrapperStyle={{
                                paddingTop: "10px",
                                fontFamily: "Inter, sans-serif",
                                fontWeight: 600,
                            }}
                        />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
