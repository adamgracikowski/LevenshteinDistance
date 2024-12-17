#pragma once

namespace Timers
{
	/// <summary>
	/// Abstract base class for implementing various types of timers.
	/// </summary>
	class Timer {
	public:
		/// <summary>
		/// This method begins tracking time from the current point.
		/// </summary>
		virtual void Start() = 0;

		/// <summary>
		/// Stops or pauses the timer.
		/// </summary>
		virtual void Stop() = 0;

		/// <summary>
		/// Retrieves the time elapsed (in milliseconds) since the last Start.
		/// </summary>
		/// <returns>Elapsed time in milliseconds for the last measurement.</returns>
		virtual float ElapsedMiliseconds() = 0;

		/// <summary>
		/// Retrieves the total accumulated time in milliseconds.
		/// </summary>
		/// <returns>Total accumulated elapsed time in milliseconds.</returns>
		virtual float TotalElapsedMiliseconds() = 0;

		/// <summary>
		/// Resets the timer to its initial state.
		/// </summary>
		virtual void Reset() = 0;

		virtual ~Timer() = default;
	};
}