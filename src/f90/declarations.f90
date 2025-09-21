
! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

MODULE params

    IMPLICIT NONE
    DOUBLE PRECISION, PARAMETER :: LAM = (1.0D0 + SQRT(5.0D0)) / 2.0D0
    DOUBLE PRECISION, PARAMETER :: AA  =   1.0D0
    DOUBLE PRECISION, PARAMETER :: BB  =   AA/LAM - AA
    DOUBLE PRECISION, PARAMETER :: CC  = - AA/LAM

    COMPLEX, PARAMETER :: im = (0.0D0,1.0D0)

END MODULE params

! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

MODULE phys

    IMPLICIT NONE

    INTEGER :: nn
    DOUBLE PRECISION :: nu

END MODULE phys

! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

MODULE vars

    IMPLICIT NONE

    DOUBLE PRECISION, ALLOCATABLE, DIMENSION (:) :: kk, k2

END MODULE vars

! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

MODULE frc

    IMPLICIT NONE

    INTEGER :: dwn, upp
    DOUBLE PRECISION :: eps

    DOUBLE PRECISION :: e0

END MODULE frc

! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

MODULE rand

    IMPLICIT NONE

    INTEGER :: seed = 1234567

END MODULE rand

! ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- |

MODULE fmt
CHARACTER(len=10) :: ext
CHARACTER(len=8) :: fmtext = '(i10.10)'
    
END MODULE fmt