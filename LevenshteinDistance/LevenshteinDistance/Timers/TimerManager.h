#pragma once

#include "HostTimer.h"
#include "DeviceTimer.cuh"

namespace Timers
{
	/// <summary>
	/// A singleton class to manage and organize timers for measuring performance.
	/// </summary>
	class TimerManager {
	private:
		TimerManager() = default;
		TimerManager(const TimerManager&) = delete;
		TimerManager& operator=(const TimerManager&) = delete;

		~TimerManager() {
			Host2DeviceDataTransferTimer.Reset();
			Device2HostDataTransferTimer.Reset();
			PopulateDeviceXTimer.Reset();
			PopulateDeviceDistancesTimer.Reset();
		}

	public:
		/// <summary>
		/// Timer for measuring the duration of loading data from the input file.
		/// </summary>
		HostTimer LoadDataFromInputFileTimer{};

		/// <summary>
		/// Timer for measuring the duration of saving data to the output file.
		/// </summary>
		HostTimer SaveDataToOutputFileTimer{};

		/// <summary>
		/// Timer for measuring the time taken to compute the distance (e.g., Levenshtein Distance).
		/// </summary>
		HostTimer FindDistanceTimer{};

		/// <summary>
		/// Timer for measuring the time taken to retrieve the transformation (edit sequence).
		/// </summary>
		HostTimer RetrieveTransformationTimer{};

		/// <summary>
		/// Timer for measuring the duration of data transfer from host (CPU) to device (GPU).
		/// </summary>
		DeviceTimer Host2DeviceDataTransferTimer{};

		/// <summary>
		/// Timer for measuring the duration of data transfer from device (GPU) to host (CPU).
		/// </summary>
		DeviceTimer Device2HostDataTransferTimer{};

		/// <summary>
		/// Timer for measuring the time taken to populate the 'X' data on the device.
		/// </summary>
		DeviceTimer PopulateDeviceXTimer{};

		/// <summary>
		/// Timer for measuring the time taken to populate distance-related data on the device.
		/// </summary>
		DeviceTimer PopulateDeviceDistancesTimer{};

		static TimerManager& GetInstance() {
			static TimerManager instance;
			return instance;
		}
	};
}