; NSIS script for the Windows installer. Built by build.sh, which passes:
;   VERSION  app version, STAGE  folder with python\, app.ico, LICENSE.txt,  OUTFILE  setup .exe
;
; Installs per user (no administrator rights), like most apps that update themselves
; or ship their own runtime: %LOCALAPPDATA%\Programs\Backing Track Generator.

Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!define APP_NAME "Backing Track Generator"
!define APP_KEY "BackingTrackGenerator"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_KEY}"
!define LAUNCH_EXE "$INSTDIR\python\pythonw.exe"
!define LAUNCH_ARGS "-m backing_track_generator"

Name "${APP_NAME}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\${APP_NAME}"
InstallDirRegKey HKCU "Software\${APP_KEY}" "InstallDir"
RequestExecutionLevel user
BrandingText "${APP_NAME} ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${APP_NAME}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "${APP_NAME} installer"
VIAddVersionKey "LegalCopyright" "MIT License"

!include "MUI2.nsh"

!define MUI_ICON "${STAGE}\app.ico"
!define MUI_UNICON "${STAGE}\app.ico"
!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_NODESC
!define MUI_FINISHPAGE_RUN "${LAUNCH_EXE}"
!define MUI_FINISHPAGE_RUN_PARAMETERS "${LAUNCH_ARGS}"
!define MUI_FINISHPAGE_RUN_TEXT "Start ${APP_NAME}"

!insertmacro MUI_PAGE_LICENSE "${STAGE}\LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "${APP_NAME}" SecApp
    SectionIn RO
    SetOutPath "$INSTDIR"
    ; An upgrade replaces the whole runtime, so no files of older packages linger.
    RMDir /r "$INSTDIR\python"
    File /r "${STAGE}\python"
    File "${STAGE}\app.ico"
    File "${STAGE}\LICENSE.txt"
    WriteUninstaller "$INSTDIR\uninstall.exe"

    CreateShortcut "$SMPROGRAMS\${APP_NAME}.lnk" "${LAUNCH_EXE}" "${LAUNCH_ARGS}" "$INSTDIR\app.ico" 0

    WriteRegStr HKCU "Software\${APP_KEY}" "InstallDir" "$INSTDIR"
    WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
    WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\app.ico"
    WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
    WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
    WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
    WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "Desktop shortcut" SecDesktop
    SetOutPath "$INSTDIR"
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "${LAUNCH_EXE}" "${LAUNCH_ARGS}" "$INSTDIR\app.ico" 0
SectionEnd

; Separated stems, settings and downloaded models are user data in other folders
; (%LOCALAPPDATA%\backing-track-generator, %APPDATA%\backing-track-generator, ~\.cache):
; they are kept, so a reinstall doesn't have to separate every song again.
Section "Uninstall"
    Delete "$SMPROGRAMS\${APP_NAME}.lnk"
    Delete "$DESKTOP\${APP_NAME}.lnk"
    ; Only what the installer put there: never a recursive delete of $INSTDIR itself.
    RMDir /r "$INSTDIR\python"
    Delete "$INSTDIR\app.ico"
    Delete "$INSTDIR\LICENSE.txt"
    Delete "$INSTDIR\uninstall.exe"
    RMDir "$INSTDIR"
    DeleteRegKey HKCU "${UNINST_KEY}"
    DeleteRegKey HKCU "Software\${APP_KEY}"
SectionEnd
