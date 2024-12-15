#pragma once

#include "HostTimer.h"

namespace Timers
{
	class TimerManager {
	private:
		TimerManager() = default;
		TimerManager(const TimerManager&) = delete;
		TimerManager& operator=(const TimerManager&) = delete;

		~TimerManager() {
			// reset timers
		}

	public:
		HostTimer LoadDataFromInputFileTimer{};
		HostTimer SaveDataToOutputFileTimer{};
		HostTimer FindDistanceTimer{};
		HostTimer RetrieveTransformationTimer{};

		static TimerManager& GetInstance() {
			static TimerManager instance;
			return instance;
		}
	};
}