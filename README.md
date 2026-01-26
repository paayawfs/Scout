# NextGen Scout 🔍⚽

**AI-powered football player scouting and similarity analysis.**

Find statistically similar football players based on comprehensive performance metrics including goals, assists, xG, passing, dribbling, defensive actions, and more.

## Features

- 🤖 **AI Player Analysis** - Gemini-powered scouting reports with tactical insights
- 📊 **Position-Specific Percentiles** - Compare FWs vs FWs, MFs vs MFs, DFs vs DFs
- 📈 **Comprehensive Stats** - Data averaged from 2024/25 and 2025/26 seasons
- 🎯 **Smart Similarity** - Find statistical twins across 7 European leagues
- 📉 **Radar Charts** - Visual comparison of player attributes
- 🔍 **Advanced Filters** - Search by position, nationality, league, and age

## Tech Stack

- **Frontend**: Next.js 16, React 19, TailwindCSS 4
- **Backend**: Supabase (PostgreSQL)
- **AI**: Google Gemini (player analysis)
- **Data**: FBref player statistics (7 leagues)

## Project Structure

```
├── web/                    # Next.js web application
│   ├── app/               # App router pages
│   └── components/        # React components
├── upload_to_supabase.py  # Database upload script
└── data/                  # Player data (gitignored)
```

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- Supabase account

### Installation

1. Clone the repository
```bash
git clone https://github.com/paayawfs/Scout.git
cd Scout
```

2. Install Python dependencies
```bash
pip install torch pandas numpy scikit-learn python-dotenv supabase tqdm
```

3. Install web dependencies
```bash
cd web
npm install
```

4. Set up environment variables
```bash
# Create .env file with:
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key

# Create web/.env.local with:
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
GEMINI_API_KEY=your_gemini_api_key
```

### Running the App

```bash
cd web
npm run dev
```


## Data Pipeline

1. **Process Data**: `python process_merged_data.py`
2. **Upload to Supabase**: `python upload_to_supabase.py`

## License

MIT License

---

*Data sourced from FBref.*
