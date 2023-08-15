venv\Scripts\activate

cmake -S "." -B "build" -DIN_PLACE_BUILD:BOOL=ON -DIS_PYTHON_BUILD:BOOL=OFF ^
  -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON ^
  -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo ^
  -DENABLE_DOCUMENTATION:BOOL=OFF ^
  -DENABLE_GITEE:BOOL=ON -DENABLE_LOGGING:BOOL=OFF ^
  -DENABLE_LOGGING_DEBUG_LEVEL:BOOL=OFF -DENABLE_LOGGING_TRACE_LEVEL:BOOL=OFF ^
  -DBUILD_TESTING:BOOL=OFF -DCLEAN_3RDPARTY_INSTALL_DIR:BOOL=OFF ^
  -DUSE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_FIND_USE_PACKAGE_REGISTRY:BOOL=OFF ^
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY:BOOL=OFF -DJOBS:STRING=4 ^
  -G "MinGW Makefiles"

rem python -m build --unset BUILD_TESTING --unset CLEAN_3RDPARTY_INSTALL_DIR ^
rem   --set ENABLE_GITEE --unset ENABLE_LOGGING --unset ENABLE_LOGGING_DEBUG_LEVEL ^
rem   --set USE_VERBOSE_MAKEFILE --var JOBS 12 build_ext --jobs 12