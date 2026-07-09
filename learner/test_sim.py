import animals_simulation

def main():
    print("Successfully imported animals_simulation!")
    sim = animals_simulation.Simulation()
    obs = sim.reset()
    print(f"Initial observation: {obs}")
    obs, reward, terminated, truncated = sim.step(1)
    print(f"Step 1 - Obs: {obs}, Reward: {reward}, Terminated: {terminated}")

if __name__ == "__main__":
    main()
