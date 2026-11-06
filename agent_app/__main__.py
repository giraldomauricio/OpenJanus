import uvicorn

if __name__ == "__main__":
    # Bound to loopback on purpose: this app can run LLM-written code once approved.
    uvicorn.run("agent_app.main:app", host="127.0.0.1", port=8000)
