# NextGen Scout 🔍⚽

**AI-powered player scouting and similarity analysis using Siamese Neural Networks.**

Find statistically similar football players based on comprehensive performance metrics including goals, assists, xG, passing, dribbling, defensive actions, and more.

## Features

- 🧠 **Siamese Neural Network** - Deep learning model trained on 100+ player statistics
- 📊 **Comprehensive Stats** - Data averaged from 2024/25 and 2025/26 seasons
- 🎯 **Smart Similarity** - Gaussian RBF kernel for accurate player matching
- 📈 **Radar Charts** - Visual comparison of player attributes
- 🔍 **Advanced Filters** - Search by position, nationality, league, and age

## Tech Stack

- **Frontend**: Next.js 16, React, TailwindCSS
- **Backend**: Supabase (PostgreSQL)
- **ML**: PyTorch, Siamese Neural Networks
- **Data**: FBref player statistics

## Project Structure

```
├── web/                    # Next.js web application
│   ├── app/               # App router pages
│   └── components/        # React components
├── train_snn_full.py      # Full-feature SNN training script
├── upload_to_supabase.py  # Database upload script
├── export_to_json.py      # Data export utilities
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
```

### Running the App

```bash
cd web
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000)

## Data Pipeline

1. **Train SNN Model**: `python train_snn_full.py`
2. **Upload to Supabase**: `python upload_to_supabase.py`

## License

MIT License

---

*Data sourced from FBref. Powered by Siamese Neural Networks.*
