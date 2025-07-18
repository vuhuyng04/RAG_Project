# RAG_Project

graph TD
    %% Data Sources
    A[🌐 Website: hoatuoimymy.com] --> B[🕷️ Web Crawler]
    
    %% Data Processing Pipeline
    B --> C[📄 Raw Content<br/>HTML, Text, Images]
    C --> D[🔄 Text Processing<br/>Clean, Chunk, Normalize]
    D --> E[🧠 Text Embedding<br/>Vector Model]
    E --> F[📊 Qdrant Vector DB<br/>Store Embeddings]
    
    %% User Query Pipeline
    G[👤 User Query] --> H[🔍 Query Processing<br/>Clean & Embed]
    H --> I[🎯 Vector Search<br/>Similarity Search]
    I --> F
    F --> J[📋 Retrieved Context<br/>Top-K Similar Documents]
    
    %% LLM Generation Pipeline
    J --> K[📝 Prompt Engineering<br/>Context + Query + Instructions]
    K --> L[🤖 LLM Model<br/>Gemini/GPT/Claude]
    L --> M[💬 Generated Response]
    
    %% User Interface
    M --> N[📱 Output Message<br/>Final Answer to User]
    N --> G
    
    %% Styling
    classDef website fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef crawler fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef processing fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef vectordb fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef llm fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef user fill:#f1f8e9,stroke:#689f38,stroke-width:2px
    
    class A website
    class B crawler
    class C,D,E processing
    class F,I,J vectordb
    class G,H,N user
    class K,L,M llm
    
    %% Subgraphs for better organization
    subgraph "📊 Data Ingestion Pipeline"
        A
        B
        C
        D
        E
    end
    
    subgraph "🔍 Query & Retrieval Pipeline"
        G
        H
        I
        J
    end
    
    subgraph "🤖 Generation Pipeline"
        K
        L
        M
        N
    end
