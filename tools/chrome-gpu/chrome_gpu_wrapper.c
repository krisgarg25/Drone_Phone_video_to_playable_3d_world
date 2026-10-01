/* Launch HyperFrames' headless Chrome on the high-performance (NVIDIA) GPU.

   Laptops hand Chrome the integrated GPU by default, so renders run on the iGPU while the
   RTX sits idle. HyperFrames has no hook for extra Chrome flags, but it takes a browser path
   (PRODUCER_HEADLESS_SHELL_PATH). This wrapper is that path: it runs the real
   chrome-headless-shell with --force_high_performance_gpu added, passes stdio and the
   inherited handles through unchanged, and puts Chrome in a kill-on-close job so it never
   outlives the wrapper.

   Build:  gcc -O2 -o chrome-headless-shell.exe chrome_gpu_wrapper.c
   Real binary: CHROME_GPU_REAL env var, else the path below. */
#include <windows.h>
#include <wchar.h>
#include <stdlib.h>

#define DEFAULT_REAL L"C:\\Users\\krisg\\.cache\\hyperframes\\chrome\\chrome-headless-shell\\win64-152.0.7977.30\\chrome-headless-shell-win64\\chrome-headless-shell.exe"

static const wchar_t *skip_argv0(const wchar_t *p) {
    if (*p == L'"') {
        p++;
        while (*p && *p != L'"') p++;
        if (*p) p++;
    } else {
        while (*p && *p != L' ' && *p != L'\t') p++;
    }
    while (*p == L' ' || *p == L'\t') p++;
    return p;
}

int main(void) {
    wchar_t real[MAX_PATH];
    if (!GetEnvironmentVariableW(L"CHROME_GPU_REAL", real, MAX_PATH)) wcscpy(real, DEFAULT_REAL);

    const wchar_t *rest = skip_argv0(GetCommandLineW());
    size_t n = wcslen(real) + wcslen(rest) + 64;
    wchar_t *cmd = (wchar_t *)malloc(n * sizeof(wchar_t));
    if (!cmd) return 1;
    _snwprintf(cmd, n, L"\"%ls\" --force_high_performance_gpu %ls", real, rest);

    /* same startup info as ours, so handles Node passed (stdio and any extra pipes) carry over */
    STARTUPINFOW si;
    GetStartupInfoW(&si);
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    HANDLE job = CreateJobObjectW(NULL, NULL);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li;
    ZeroMemory(&li, sizeof(li));
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));

    if (!CreateProcessW(real, cmd, NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        free(cmd);
        return 1;
    }
    AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    free(cmd);
    return (int)code;
}
