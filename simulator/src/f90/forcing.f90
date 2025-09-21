! --------------------------------------------------- |
! forcing                                             |

SUBROUTINE forcing(uu, ff)

    USE params
    USE phys
    USE rand
    USE frc

    IMPLICIT NONE

    DOUBLE COMPLEX, DIMENSION(nn), INTENT(IN)  :: uu
    DOUBLE COMPLEX, DIMENSION(nn), INTENT(OUT) :: ff

    DOUBLE PRECISION :: phs, tmp
    INTEGER :: i

    tmp = 0.0D0
    DO i = dwn, upp
        tmp = tmp + abs(uu(i))**2    ! Calculate total energy in forcing range
    END DO
    tmp = eps / tmp                   ! Scale factor based on energy

    CALL seedInit(seed)              ! Initialize random number generator
    DO i = dwn, upp
        CALL randuni(phs)         ! Get random phase
        tmp = tmp + im*phs        ! Add imaginary component with random phase
        ff(i) = uu(i)*tmp         ! Apply forcing to mode i, note that forcing is applied only to mode from dwn to upp
    END DO

END SUBROUTINE forcing