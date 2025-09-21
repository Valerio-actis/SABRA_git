! --------------------------------------------------- |
! initial conditions                                  |

SUBROUTINE initial(uu)

    USE params
    USE phys
    USE rand
    USE frc

    IMPLICIT NONE

    DOUBLE COMPLEX, DIMENSION(nn), INTENT(OUT) :: uu

    DOUBLE PRECISION :: phs
    DOUBLE PRECISION :: tmp
    INTEGER :: i

    DO i = 1, nn
        uu(i) = 0.0D0
    END DO

    CALL seedInit(seed)
    DO i = dwn, upp
        CALL randuni(phs)
        uu(i) = exp(im * phs)
    END DO

    tmp = 0.0D0
    DO i = dwn, upp
        tmp = tmp + abs(uu(i))**2
    END DO

    uu = SQRT(e0) * uu / SQRT(tmp)

END SUBROUTINE initial
