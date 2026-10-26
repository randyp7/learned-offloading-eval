from pathlib import Path
from typing import Tuple, Optional
import re


class BerlinDataResolver:
    """
    Resolves Berlin traffic data by date and optional timeframe

    Only uses date-based SUMO configs (24-hour simulations).
    If timeframe is specified, returns the config with calculated begin/end times.

    Supported pattern:
    - berlin_YYYY-MM-DD.sumocfg (full 24-hour day)
    """

    def __init__(self, project_root: Path = None):
        if project_root is None:
            # Default to project root (4 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent
        self.project_root = project_root
        self.sumo_dir = project_root / "sumo"
        self._build_index()

    def _build_index(self):
        """Build index of available date-based SUMO configs"""
        self.by_date = {}  # date -> Path

        if not self.sumo_dir.exists():
            return

        # Only search in sumo root directory (configs there have correct paths)
        for config_file in self.sumo_dir.glob("*.sumocfg"):
            self._index_file(config_file)

    def _index_file(self, config_file: Path):
        """Parse and index a single config file - only date-based configs"""
        stem = config_file.stem

        # Only index date-based configs (ignore timeframe-specific like berlin_a100_*)
        # Pattern: berlin_YYYY-MM-DD
        date_match = re.search(r'berlin_(\d{4}-\d{2}-\d{2})$', stem)
        if date_match:
            date = date_match.group(1)
            self.by_date[date] = config_file

    def resolve(
        self,
        date: str,
        timeframe: Optional[Tuple[int, int]] = None
    ) -> dict:
        """
        Get SUMO config for date and optional timeframe

        Args:
            date: Date string in format "YYYY-MM-DD"
            timeframe: Optional (start_hour, end_hour) e.g., (7, 9) to restrict simulation

        Returns:
            dict with:
                - 'config_path': Path to SUMO config file
                - 'begin_time': Begin time in seconds (0 or calculated from timeframe)
                - 'end_time': End time in seconds (86400 or calculated from timeframe)

        Raises:
            FileNotFoundError if no matching data
            ValueError if date is None
        """
        if not date:
            raise ValueError("Date must be specified")

        if date not in self.by_date:
            error_msg = f"No Berlin SUMO config found for date={date}\n\n"
            error_msg += f"Available dates: {sorted(self.by_date.keys())[:20]}"
            if len(self.by_date) > 20:
                error_msg += f"\n... and {len(self.by_date) - 20} more"
            raise FileNotFoundError(error_msg)

        config_path = self.by_date[date]

        # Calculate begin/end times based on timeframe
        if timeframe:
            start_hour, end_hour = timeframe
            if not (0 <= start_hour < 24 and 0 < end_hour <= 24 and start_hour < end_hour):
                raise ValueError(
                    f"Invalid timeframe {timeframe}. "
                    f"Must have 0 <= start < 24, 0 < end <= 24, and start < end"
                )
            begin_time = start_hour * 3600  # Convert hours to seconds
            end_time = end_hour * 3600
        else:
            # Use full 24-hour simulation
            begin_time = 0
            end_time = 86400  # 24 hours in seconds

        return {
            'config_path': config_path,
            'begin_time': begin_time,
            'end_time': end_time
        }

    def list_available(self) -> dict:
        """List all available date-based configurations"""
        return {
            'dates': sorted(self.by_date.keys()),
            'total_count': len(self.by_date)
        }


# Singleton
_resolver = None

def get_resolver(project_root: Path = None) -> BerlinDataResolver:
    """Get or create the singleton resolver"""
    global _resolver
    if _resolver is None:
        _resolver = BerlinDataResolver(project_root)
    return _resolver

def resolve_berlin_data(
    date: str,
    timeframe: Optional[Tuple[int, int]] = None,
    project_root: Path = None
) -> dict:
    """
    Get SUMO config for date and optional timeframe

    Args:
        date: Date string "YYYY-MM-DD"
        timeframe: Optional (start_hour, end_hour) tuple to restrict simulation
        project_root: Project root path (optional, auto-detected if None)

    Returns:
        dict with 'config_path', 'begin_time', and 'end_time'
    """
    resolver = get_resolver(project_root)
    return resolver.resolve(date=date, timeframe=timeframe)

def list_available_berlin_data(project_root: Path = None) -> dict:
    """List all available Berlin SUMO configurations"""
    resolver = get_resolver(project_root)
    return resolver.list_available()
