#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);

    char *lastSlash = strrchr(exePath, '\\');
    if (!lastSlash) lastSlash = strrchr(exePath, '/');
    if (lastSlash) *(lastSlash + 1) = '\0';

    char realFfmpeg[MAX_PATH];
    snprintf(realFfmpeg, MAX_PATH, "%sffmpeg_real.exe", exePath);

    size_t cmdBufSize = 65536;
    char *newCmd = (char *)malloc(cmdBufSize);
    if (!newCmd) return 1;

    snprintf(newCmd, cmdBufSize, "\"%s\"", realFfmpeg);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-vsync") == 0) {
            strncat(newCmd, " -fps_mode", cmdBufSize - strlen(newCmd) - 1);
            if (i + 1 < argc && strcmp(argv[i + 1], "0") == 0) {
                strncat(newCmd, " passthrough", cmdBufSize - strlen(newCmd) - 1);
                i++;
            }
        } else {
            strncat(newCmd, " ", cmdBufSize - strlen(newCmd) - 1);
            int hasSpace = (strchr(argv[i], ' ') != NULL || strchr(argv[i], '\t') != NULL);
            if (hasSpace) strncat(newCmd, "\"", cmdBufSize - strlen(newCmd) - 1);
            strncat(newCmd, argv[i], cmdBufSize - strlen(newCmd) - 1);
            if (hasSpace) strncat(newCmd, "\"", cmdBufSize - strlen(newCmd) - 1);
        }
    }

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessA(NULL, newCmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "[ffmpeg_wrapper] Failed to launch ffmpeg_real.exe (error %lu)\n", GetLastError());
        free(newCmd);
        return 1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    free(newCmd);

    return (int)exitCode;
}
