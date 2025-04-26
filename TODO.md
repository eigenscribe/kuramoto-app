- [ ] Ensure good time step usage
   <details>
      <summary>✨Chat Aid Advice</summary>
      <ul>
         What is the "time step" in numerical simulations?
         In numerical simulations, the "time step" $\Delta t$ is the small increment of time used to advance the solution of your system from one point to the next. For the Kuramoto Model, which involves solving a system of coupled differential equations, the time step determines how accurately and stably the system evolves over time.

      </ul>
      
      If the time step is too large:
      - The simulation might become unstable or inaccurate.
      - You might miss important dynamics of the system.

      If the time step is too small:
      - The simulation will be very slow because it takes many small steps to cover the same amount of time.
      - You might waste computational resources without gaining much accuracy.

      ### What does a "time step optimizer" button do?
      I assume your button is meant to help users automatically choose a good time step for the simulation. To determine if it's working correctly, we need to consider:
      1. **Accuracy**: Does the chosen time step produce results that are close to the true solution?
      2. **Stability**: Does the simulation remain stable (i.e., it doesn't blow up or produce nonsensical results)?
      3. **Efficiency**: Does the chosen time step balance speed and accuracy?

      ### How to check if your button is valid
      Here are some steps you can take to validate your time step optimizer:

      #### 1. Test with known solutions
      The Kuramoto Model has some cases where the behavior is well understood. For example:
      - If all oscillators are identical and fully synchronized, the system evolves predictably.
      - If the coupling strength is very weak, the oscillators behave almost independently.

      Run your simulation with these simple cases and compare the results to the expected behavior. If the time step optimizer chooses a step size that produces accurate results, it's likely working well.

      #### 2. Check for stability
      Run your simulation with a range of time steps (both smaller and larger than the one chosen by your optimizer). If the simulation becomes unstable (e.g., values blow up or oscillate wildly) for larger time steps, but remains stable for the one chosen by your optimizer, that's a good sign.

      #### 3. Compare with smaller time steps
      Run the simulation with a much smaller time step (e.g., half or one-tenth of the one chosen by your optimizer) and compare the results. If the results are very similar, it means the chosen time step is accurate enough. If there are large differences, your optimizer might be choosing a time step that's too large.

      #### 4. Look at computational efficiency
      Measure how long the simulation takes with the time step chosen by your optimizer versus a much smaller time step. If the optimizer's time step is significantly faster while still being accurate, it's doing its job well.

      ### Possible improvements
      If you find that your time step optimizer isn't working as well as you'd like, here are some ideas for improvement:

      1. **Adaptive time stepping**:
         Instead of using a fixed time step, you could implement an adaptive time-stepping method. This adjusts the time step dynamically based on the system's behavior. For example:
         - If the oscillators are moving slowly, you can use a larger time step.
         - If the oscillators are moving quickly or the system is highly coupled, you can use a smaller time step.

         Common methods for adaptive time stepping include Runge-Kutta-Fehlberg (RKF) or Dormand-Prince (used in MATLAB's `ode45`).

      2. **Error estimation**:
         Use a numerical method that provides an estimate of the error at each step (e.g., embedded Runge-Kutta methods). If the error is too large, reduce the time step; if the error is very small, increase the time step.

      3. **User feedback**:
         Provide feedback to the user about the chosen time step. For example, display a message like "Time step optimized for accuracy and efficiency" or "Warning: Time step may be too large for highly coupled systems."

      4. **Parameter sensitivity**:
         The optimal time step might depend on parameters like the coupling strength (\(K\)) or the number of oscillators. You could include logic in your optimizer to adjust the time step based on these parameters.

      ### Example of a simple time step optimizer
      If you're using a fixed time step, you could base it on the system's parameters. For example:
      - If the natural frequencies of the oscillators (\(\omega_i\)) are large, use a smaller time step.
      - If the coupling strength (\(K\)) is large, use a smaller time step.

      A simple heuristic might look like this:
      ```python
      max_frequency = max(abs(omega))  # Largest natural frequency
      time_step = 1 / (10 * max_frequency)  # Choose a time step that's small enough to resolve the fastest oscillator
      ```
   </details>

- [ ] Animation tab: Put buttons on top of time point slide bar
- [ ] Put time progression bar under time point slider
- [ ] Make 'fill' color on time progression bar green
- [ ] Sidebar: put time controls  above random seed parameter
- [ ] Sidebar: make style of parameter headings have the same blue gradient as title
- [ ] Sidebar: Increase maximum simulation time to 200
- [ ] Database implementation