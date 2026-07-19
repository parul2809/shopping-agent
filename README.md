# 🛒 AI Shopping Agent

An AI-powered shopping assistant built with LangGraph and Groq. It searches products, checks ratings, and places orders through a conversational Streamlit interface.

## Features

- **Natural language shopping** — describe what you want and the agent finds matching products
- **Smart filtering** — filter by price, organic status, and minimum rating
- **Image search** — upload a product photo to find similar items in the store
- **One-click checkout** — confirm and the agent places your order

## Tech Stack

- **LangGraph** — ReAct agent with tool calling
- **Groq** — Fast LLM inference (Llama 3.1 8B)
- **Streamlit** — Chat UI
- **SQLite** — Product catalog, reviews, and orders

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/parul2809/shopping-agent.git
cd shopping-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Get a free API key from [Groq Console](https://console.groq.com/keys), then create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```
GROQ_API_KEY="gsk_your_key_here"
```

### 5. Initialize the database

```bash
python setup_db.py
```

### 6. Run the app

```bash
streamlit run app.py
```

## Project Structure

```
shopping-agent/
├── app.py              # Streamlit chat UI
├── shopping_agent.py   # LangGraph agent + tools
├── reviews_api.py      # Product rating API
├── setup_db.py         # Database initialization script
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
└── README.md
```

## Usage Examples

- "I want organic honey under $15 with 4+ rating"
- "Show me coffee options"
- "Find me the best rated nuts"
- Upload a product image to find similar items

## Deploy to Streamlit Community Cloud

1. **Push to GitHub** (make sure `.env` is NOT committed):
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. **Click "New app"** and select:
   - Repository: `YOUR_USERNAME/shopping-agent`
   - Branch: `main`
   - Main file path: `app.py`

4. **Add your secret** — In the app's settings, go to **Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

5. **Deploy** — Click "Deploy" and your app will be live at `https://your-app-name.streamlit.app`

## License

MIT
