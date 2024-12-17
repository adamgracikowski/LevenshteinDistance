#pragma once

#include "HostTimer.h"
#include "DeviceTimer.cuh"

namespace Timers
{
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
		HostTimer LoadDataFromInputFileTimer{};
		HostTimer SaveDataToOutputFileTimer{};
		HostTimer FindDistanceTimer{};
		HostTimer RetrieveTransformationTimer{};

		DeviceTimer Host2DeviceDataTransferTimer{};
		DeviceTimer Device2HostDataTransferTimer{};
		DeviceTimer PopulateDeviceXTimer{};
		DeviceTimer PopulateDeviceDistancesTimer{};

		static TimerManager& GetInstance() {
			static TimerManager instance;
			return instance;
		}
	};
}