import asyncio
from client import MyHackathonEnv
from server.models import MyHackathonAction

async def main():
    # 1. Connect to your local server
    env = MyHackathonEnv(base_url="http://localhost:8000")
    
    # 2. Reset the environment to start the game
    print("--- Starting New Game ---")
    result = await env.reset()
    print(f"Problem: {result.observation.problem}")
    
    # 3. Play for 3 steps
    for i in range(3):
        # In a real hackathon, an AI would choose this. 
        # For now, let's just send a guess.
        # Check the problem text and try to answer it!
        ans = int(input("Your Answer: "))
        
        action = MyHackathonAction(answer=ans)
        result = await env.step(action)
        
        print(f"Feedback: {result.observation.message}")
        print(f"Reward: {result.reward} | Level: {result.observation.current_level}")
        
        if not result.done:
            print(f"Next Problem: {result.observation.problem}")
        else:
            print("Game Over!")
            break

if __name__ == "__main__":
    asyncio.run(main())