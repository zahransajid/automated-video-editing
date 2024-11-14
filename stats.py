from pstats import Stats, SortKey


stats = Stats("video_details/profile_results").strip_dirs().sort_stats(SortKey.TIME).print_stats()