## Implementation for Language Model Self-play via Scorable Negotiation Game

### Usage

1. **Setup the environment:**

   python>=3.11,
   cuda==11.8,
   torch==2.7.1cu118

   ```
   pip install -r requirements.txt
   ```

3. **Download DeepSeek-R1-Distill-Qwen-7B from hugging face:**

   ```
   export HF_ENDPOINT=https://hf-mirror.com
   
   huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./DeepSeek-R1-Distill-Qwen-7B --local-dir-use-symlinks False --resume-download --token <your_huggingface_token>
   ```

4. **Reproduce the Results from the Paper:**

   **Step 1: start generalization-aware behavioral cloning**

   ```
   bash sft.sh
   ```

   **Step 2: start self-play**

   ```
   bash play_game.sh
   ```

   **Step 3: assign rewards**

   ```
   bash assign_rewards.sh
   ```

   **Step 4: start our training**

   ```
   bash lssg.sh
   ```

5. **Evaluate the Results of Our Model:**

   **Reasoning Evaluation:**

   - Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for reasoning evaluation

   **Negotiation Evaluation:**

   - Use [NegotiationArena](https://github.com/vinid/NegotiationArena) for negotiation evaluation
