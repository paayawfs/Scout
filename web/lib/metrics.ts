
export const METRIC_SETS: Record<string, { key: string; label: string; angle: number }[]> = {
    // Forwards / Attackers
    'FW': [
        { key: 'Non-Penalty xG', label: 'Non-Pen xG', angle: 0 },
        { key: 'Goals', label: 'Goals', angle: 60 },
        { key: 'xAG', label: 'xAG', angle: 120 },
        { key: 'Key Passes', label: 'Key Passes', angle: 180 },
        { key: 'Successful Dribbles', label: 'Dribbles', angle: 240 },
        { key: 'Progressive Carries', label: 'Carries', angle: 300 },
    ],
    // Midfielders (Balanced)
    'MF': [
        { key: 'Key Passes', label: 'Key Passes', angle: 0 },
        { key: 'Progressive Passes', label: 'Prog Passes', angle: 60 },
        { key: 'Interceptions', label: 'Interceptions', angle: 120 },
        { key: 'Tackles', label: 'Tackles', angle: 180 },
        { key: 'Assists', label: 'Assists', angle: 240 },
        { key: 'xAG', label: 'xAG', angle: 300 },
    ],
    // Defenders
    'DF': [
        { key: 'Tackles', label: 'Tackles', angle: 0 },
        { key: 'Interceptions', label: 'Interceptions', angle: 60 },
        { key: 'Tackles Won', label: 'Tkl Won', angle: 120 },
        { key: 'Progressive Passes', label: 'Prog Passes', angle: 180 },
        { key: 'Progressive Carries', label: 'Carries', angle: 240 },
        { key: 'Key Passes', label: 'Key Passes', angle: 300 },
    ]
};

export const getPositionSet = (pos: string) => {
    if (!pos) return METRIC_SETS['MF'];
    if (pos.includes('FW') || pos.includes('AM')) return METRIC_SETS['FW'];
    if (pos.includes('DF') || pos.includes('CB') || pos.includes('FB')) return METRIC_SETS['DF'];
    return METRIC_SETS['MF'];
};
